"""Customer Intelligence: upload, prepare, segment, inspect, and export customer data."""

from __future__ import annotations

import hashlib
import io
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_TRAIN_ROWS = 20_000
SILHOUETTE_SAMPLE = 3_000
MAX_FILE_MB = 50
RANDOM_STATE = 42
NONE_LABEL = "— Not mapped —"

ALIASES = {
    "customer_id": [
        "customer_unique_id", "customer_id", "customerid", "client_id",
        "clientid", "buyer_id", "member_id", "user_id",
    ],
    "order_id": [
        "order_id", "orderid", "invoice", "invoice_no", "invoiceno", "invoice_number",
        "transaction_id", "basket_id", "receipt_id",
    ],
    "product_id": [
        "product_name", "description", "item_name", "product", "product_id", "productid",
        "stockcode", "stock_code", "sku", "item_id", "item",
    ],
    "category": [
        "product_category_english", "product_category_name", "category",
        "product_category", "product_type", "department", "item_category",
    ],
    "date": [
        "date", "order_date", "purchase_date", "invoice_date",
        "transaction_date", "order_purchase_timestamp", "invoicedate", "created_at",
    ],
    "quantity": ["quantity", "qty", "units", "units_sold", "items_sold", "inferred_item_quantity"],
    "unit_price": ["unit_price", "price", "selling_price", "unitprice", "price_each", "item_price"],
    "revenue": [
        "line_item_revenue", "revenue", "sales", "sales_amount", "line_revenue",
        "total_amount", "amount", "total_spend", "total_item_spend", "monetary_value",
    ],
    "freight": ["freight", "shipping", "shipping_cost", "freight_value"],
}

ID_TOKENS = {"id", "identifier", "code", "postal", "zip"}
DEFAULT_FEATURE_HINTS = (
    "spend", "revenue", "sales", "amount", "order", "frequency", "quantity",
    "product", "category", "purchase_span", "recency", "freight", "unit_price",
)


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def looks_like_identifier(column: str) -> bool:
    tokens = set(normalize_name(column).split("_"))
    return bool(tokens & ID_TOKENS)


def find_alias(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {normalize_name(column): column for column in columns}
    return next((lookup[alias] for alias in aliases if alias in lookup), None)


def detect_mapping(columns: list[str]) -> dict[str, str | None]:
    return {key: find_alias(columns, aliases) for key, aliases in ALIASES.items()}


def numeric_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert common currency/comma-formatted values to numbers and count failures."""
    text = series.astype("string").str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"[$€£₹]", "", regex=True)
    converted = pd.to_numeric(text, errors="coerce")
    bad_count = int((series.notna() & converted.isna()).sum())
    converted = converted.replace([np.inf, -np.inf], np.nan)
    return converted, bad_count


def infer_product_categories(descriptions: pd.Series) -> pd.Series:
    """Assign broad, explicitly inferred categories from item description text."""
    text = descriptions.astype("string").str.lower().fillna("")
    rules = [
        ("Home & Kitchen", r"mug|cup|plate|bowl|teapot|kitchen|cook|cutlery|napkin|jar|tin|glassware"),
        ("Home Decor & Gifts", r"candle|ornament|frame|cushion|sign|gift|decoration|decoupage|heart|hanging"),
        ("Accessories & Bags", r"bag|purse|wallet|jewel|necklace|bracelet|earring|brooch|scarf|hat|umbrella"),
        ("Stationery & Crafts", r"paper|notebook|pen |pencil|card |card\b|craft|scrapbook|ribbon|sticker|stencil"),
        ("Lighting", r"lamp|light|lantern|candleholder|candlestick"),
        ("Toys & Games", r"toy|game|puzzle|playhouse|doll|skipping rope"),
        ("Garden & Outdoors", r"garden|plant|flower|bird|watering|outdoor|greenhouse"),
        ("Textiles & Soft Furnishings", r"blanket|towel|fabric|textile|quilt|doormat|bedding|cotton"),
        ("Party & Seasonal", r"party|christmas|easter|halloween|valentine|bunting|wreath|seasonal"),
        ("Food & Drink", r"tea\b|coffee|chocolate|sweets|candy|food|jam\b|biscuit|sugar"),
        ("Children & Baby", r"baby|child|children|bib\b|feeding|nursery"),
    ]
    result = pd.Series("Other / unclassified", index=descriptions.index, dtype="string")
    remaining = text.ne("")
    for label, pattern in rules:
        matched = remaining & text.str.contains(pattern, regex=True, na=False)
        result.loc[matched] = label
        remaining &= ~matched
    result.loc[text.eq("")] = "Unspecified"
    return result


def numeric_candidates(frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    candidates = []
    for column in frame.columns:
        if column in excluded:
            continue
        values, _ = numeric_series(frame[column])
        non_missing = int(frame[column].notna().sum())
        if non_missing and values.notna().sum() / non_missing >= 0.8 and values.nunique(dropna=True) > 1:
            candidates.append(column)
    return candidates


def guess_grain(frame: pd.DataFrame, mapping: dict[str, str | None]) -> str:
    customer_col = mapping["customer_id"]
    order_col = mapping["order_id"]
    if customer_col and order_col:
        customer_values = frame[customer_col].dropna()
        if customer_values.duplicated().any():
            return "Transaction-level (one row per item/order)"
    return "Customer-level (one row per customer)"


def select_mapping(label: str, columns: list[str], suggested: str | None, key: str) -> str | None:
    options = [NONE_LABEL, *columns]
    index = options.index(suggested) if suggested in columns else 0
    choice = st.selectbox(label, options, index=index, key=key)
    return None if choice == NONE_LABEL else choice


def load_uploaded_file(file_bytes: bytes, file_name: str, sheet_name: str | None) -> pd.DataFrame:
    suffix = file_name.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        raise ValueError("The CSV encoding could not be read. Save it as UTF-8 and upload again.")
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    raise ValueError("Upload a CSV or Excel file (.csv, .xlsx, or .xls).")


def transaction_to_customers(
    source: pd.DataFrame,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    customer_col = mapping["customer_id"]
    if not customer_col:
        raise ValueError(
            "A transaction-level file needs a customer identifier column so rows can be grouped. "
            "Map a customer ID column, or upload a file with one row per customer."
        )

    work = source.copy()
    quality: dict[str, object] = {"input_rows": len(source), "duplicate_rows": int(source.duplicated().sum())}
    customer_text = work[customer_col].astype("string").str.strip()
    valid_customer = customer_text.notna() & customer_text.ne("")
    quality["rows_missing_customer_id"] = int((~valid_customer).sum())
    work = work.loc[valid_customer].copy()
    work[customer_col] = customer_text.loc[valid_customer].astype(str)
    if work.empty:
        raise ValueError("No rows have a usable customer ID. Add customer IDs or use a customer-level file.")

    quantity_col = mapping["quantity"]
    price_col = mapping["unit_price"]
    revenue_col = mapping["revenue"]
    order_col = mapping["order_id"]
    product_col = mapping["product_id"]
    category_col = mapping["category"]
    inferred_category = False
    date_col = mapping["date"]
    freight_col = mapping["freight"]
    conversion_errors = {}

    for role, column in (
        ("quantity", quantity_col),
        ("unit price", price_col),
        ("revenue", revenue_col),
        ("freight", freight_col),
    ):
        if column:
            work[f"__{role.replace(' ', '_')}"], bad = numeric_series(work[column])
            conversion_errors[column] = bad
    quality["invalid_numeric_values_coerced_to_missing"] = {
        column: count for column, count in conversion_errors.items() if count
    }

    if revenue_col:
        work["__spend"] = work["__revenue"]
        spend_definition = f"Sum of mapped revenue field '{revenue_col}'"
    elif price_col and quantity_col:
        work["__spend"] = work["__unit_price"] * work["__quantity"]
        spend_definition = f"Sum of {price_col} × {quantity_col}"
    elif price_col:
        work["__spend"] = work["__unit_price"]
        spend_definition = f"Sum of mapped price field '{price_col}' (assumes each row is one item)"
    else:
        work["__spend"] = np.nan
        spend_definition = None

    if not category_col and product_col:
        work["__inferred_category"] = infer_product_categories(work[product_col])
        category_col = "__inferred_category"
        inferred_category = True
        quality["category_source"] = f"Broad category inferred from descriptions in '{product_col}'"
        quality["category_warning"] = "These are approximate text-based labels, not official product taxonomy. Review them before business decisions."

    grouped = work.groupby(customer_col, sort=False, dropna=False)
    customer = pd.DataFrame(index=grouped.size().index)
    customer.index.name = "customer_id"
    customer["transaction_rows"] = grouped.size()
    if "__spend" in work:
        customer["total_spend"] = grouped["__spend"].sum(min_count=1)
    if quantity_col:
        customer["total_quantity"] = grouped["__quantity"].sum(min_count=1)
    if price_col:
        # Separates premium-price shoppers from customers who spend more only
        # because they place more orders or buy more units.
        customer["average_unit_price"] = grouped["__unit_price"].mean()
    if order_col:
        customer["distinct_orders"] = grouped[order_col].nunique(dropna=True)
        if "total_spend" in customer:
            safe_orders = customer["distinct_orders"].replace(0, np.nan)
            customer["average_order_value"] = customer["total_spend"] / safe_orders
    if product_col:
        customer["distinct_products"] = grouped[product_col].nunique(dropna=True)
    if category_col:
        customer["distinct_categories"] = grouped[category_col].nunique(dropna=True)
        category_counts = (
            work.loc[work[category_col].notna()]
            .groupby([customer_col, category_col], sort=False, dropna=True)
            .size()
            .rename("_category_rows")
            .reset_index()
            .sort_values("_category_rows", ascending=False, kind="stable")
            .drop_duplicates(customer_col)
            .set_index(customer_col)[category_col]
        )
        customer["most_purchased_category"] = category_counts
    if freight_col:
        customer["total_freight"] = grouped["__freight"].sum(min_count=1)
    if date_col:
        parsed_dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        quality["invalid_dates"] = int((work[date_col].notna() & parsed_dates.isna()).sum())
        work["__date"] = parsed_dates
        date_group = work.groupby(customer_col, sort=False)["__date"]
        first_dates = date_group.min()
        last_dates = date_group.max()
        customer["first_purchase_date"] = first_dates
        customer["last_purchase_date"] = last_dates
        customer["purchase_span_days"] = (last_dates - first_dates).dt.total_seconds() / 86400

    # Preserve additional numeric business fields as both totals and averages.
    excluded = {
        customer_col, order_col, product_col, category_col, date_col,
        quantity_col, price_col, revenue_col, freight_col,
    }
    excluded.update(column for column in work.columns if str(column).startswith("__"))
    for column in numeric_candidates(work, {c for c in excluded if c}):
        values, bad = numeric_series(work[column])
        if bad:
            quality.setdefault("invalid_numeric_values_coerced_to_missing", {})[column] = bad
        work[f"__extra_{column}"] = values
        safe = normalize_name(column) or "measure"
        customer[f"total_{safe}"] = work.groupby(customer_col, sort=False)[f"__extra_{column}"].sum(min_count=1)
        customer[f"average_{safe}"] = work.groupby(customer_col, sort=False)[f"__extra_{column}"].mean()

    customer = customer.reset_index()
    category_report = None
    if category_col:
        category_work = work.loc[work[category_col].notna()].copy()
        if not category_work.empty:
            aggregations: dict[str, tuple[str, str]] = {"transaction_rows": (category_col, "size")}
            if "__spend" in category_work and category_work["__spend"].notna().any():
                aggregations["sales_value"] = ("__spend", "sum")
            category_report = category_work.groupby(category_col, dropna=True).agg(**aggregations)
            category_report = category_report.sort_values(
                "sales_value" if "sales_value" in category_report else "transaction_rows",
                ascending=False,
            ).reset_index(names="category")

    quality["customers_created"] = len(customer)
    quality["spend_definition"] = spend_definition
    quality["categories_inferred_from_description"] = inferred_category
    return customer, quality, category_report


def customer_level_data(
    source: pd.DataFrame,
    customer_col: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    quality: dict[str, object] = {
        "input_rows": len(source),
        "duplicate_rows": int(source.duplicated().sum()),
        "rows_missing_customer_id": 0,
    }
    customer = source.copy()
    if customer_col:
        ids = customer[customer_col].astype("string").str.strip()
        missing = ids.isna() | ids.eq("")
        quality["rows_missing_customer_id"] = int(missing.sum())
        # Preserve rows but give anonymous rows distinct labels; do not merge them.
        ids = ids.astype("object")
        ids.loc[missing] = [f"Unidentified row {i + 1}" for i in range(int(missing.sum()))]
        customer["customer_analysis_id"] = ids.astype(str)
        quality["duplicate_customer_ids"] = int(customer["customer_analysis_id"].duplicated().sum())
    else:
        customer.insert(0, "customer_analysis_id", [f"Customer {i + 1}" for i in range(len(customer))])
        quality["customer_id_note"] = "No customer ID was mapped; row numbers are used as output labels."
    quality["customers_created"] = len(customer)
    return customer, quality


def suggest_names(results: pd.DataFrame, feature_columns: list[str]) -> dict[int, str]:
    profiles = results.groupby("cluster_id")[feature_columns].median(numeric_only=True)
    normalized = {normalize_name(column): column for column in feature_columns}
    spend_candidates = [
        "total_spend", "total_item_spend", "total_revenue", "revenue",
        "sales", "amount_spent", "spend", "monetary_value",
    ]
    frequency_candidates = [
        "distinct_orders", "order_count", "purchase_frequency",
        "frequency", "orders",
    ]
    spend_col = next((normalized[c] for c in spend_candidates if c in normalized), None)
    frequency_col = next((normalized[c] for c in frequency_candidates if c in normalized), None)
    names: dict[int, str] = {}

    if frequency_col:
        frequency_mid = profiles[frequency_col].median()
        spend_mid = profiles[spend_col].median() if spend_col else None
        order_values = pd.to_numeric(results[frequency_col], errors="coerce")
        repeat_share_by_cluster = order_values.gt(1).groupby(results["cluster_id"]).mean()
        for cluster, profile in profiles.iterrows():
            frequency_value = profile[frequency_col]
            frequency = (
                float(frequency_value)
                if pd.notna(frequency_value)
                else (float(frequency_mid) if pd.notna(frequency_mid) else 0.0)
            )
            frequency_midpoint = float(frequency_mid) if pd.notna(frequency_mid) else 1.0
            repeat_share = repeat_share_by_cluster.get(cluster, 0.0)
            repeat_share = float(repeat_share) if pd.notna(repeat_share) else 0.0
            repeat = frequency > max(1.0, frequency_midpoint) or repeat_share >= 0.5
            mostly_one_time = repeat_share < 0.5
            if spend_col:
                spend_value = profile[spend_col]
                higher_spend = bool(
                    pd.notna(spend_value)
                    and pd.notna(spend_mid)
                    and float(spend_value) >= float(spend_mid)
                )
                if mostly_one_time:
                    name = "Mostly Higher-Spend One-Time Buyers" if higher_spend else "Mostly One-Time Buyers"
                elif repeat and higher_spend:
                    name = "High-Value Repeat Buyers"
                elif repeat:
                    name = "Frequent Value Shoppers"
                elif higher_spend:
                    name = "Higher-Spend Occasional Buyers"
                else:
                    name = "Occasional Buyers"
            elif mostly_one_time:
                name = "Mostly One-Time Buyers"
            else:
                name = "Frequent Repeat Buyers" if repeat else "Occasional Buyers"
            names[int(cluster)] = name
        # If two profiles share a rule-based label, distinguish them with a
        # measured relative level instead of an arbitrary numbered suffix.
        by_name: dict[str, list[int]] = {}
        for cluster, name in names.items():
            by_name.setdefault(name, []).append(cluster)
        for name, clusters in by_name.items():
            if len(clusters) < 2:
                continue
            rank_column = spend_col or frequency_col
            ordered = sorted(
                clusters,
                key=lambda cluster: (
                    float(profiles.loc[cluster, rank_column])
                    if pd.notna(profiles.loc[cluster, rank_column]) else float("-inf")
                ),
                reverse=True,
            )
            for position, cluster in enumerate(ordered):
                if position == 0:
                    modifier = "Highest Median Spend" if spend_col else "Most Frequent"
                elif position == len(ordered) - 1:
                    modifier = "Lowest Median Spend" if spend_col else "Least Frequent"
                else:
                    modifier = "Mid-Tier Median Spend" if spend_col else "Mid-Tier Frequency"
                names[cluster] = f"{modifier} {name}"
        return names

    overall = results[feature_columns].median(numeric_only=True)
    for cluster, profile in profiles.iterrows():
        valid = [
            feature for feature in feature_columns
            if pd.notna(profile[feature]) and pd.notna(overall.get(feature))
        ]
        if not valid:
            names[int(cluster)] = f"Customer Group {int(cluster) + 1}"
            continue
        strongest = max(
            valid,
            key=lambda feature: abs(float(profile[feature]) - float(overall[feature]))
            / max(abs(float(overall[feature])), 1.0),
        )
        direction = "Higher" if float(profile[strongest]) >= float(overall[strongest]) else "Lower"
        names[int(cluster)] = f"{direction} {str(strongest).replace('_', ' ').title()} Customers"
    return names


def build_summary(
    filename: str,
    grain: str,
    customer_data: pd.DataFrame,
    profile_table: pd.DataFrame,
    features: list[str],
    k: int,
    silhouette: float,
    dbi: float,
    spend_definition: str | None,
    training_rows: int,
) -> str:
    lines = [
        "CUSTOMER INTELLIGENCE — ANALYSIS SUMMARY",
        "",
        f"Uploaded file: {filename}",
        f"Data grain: {grain}",
        f"Customers analysed: {len(customer_data):,}",
        f"Features used for clustering: {', '.join(features)}",
        f"Number of clusters: {k}",
        f"Silhouette score: {silhouette:.4f}",
        f"Davies-Bouldin index: {dbi:.4f}",
        f"Rows used to fit model: {training_rows:,}",
    ]
    if spend_definition:
        lines.append(f"Sales value definition: {spend_definition}")
    lines.extend(["", "CLUSTER PROFILES"])
    for _, row in profile_table.iterrows():
        lines.append(
            f"- {row['segment_name']}: {int(row['customer_count']):,} customers "
            f"({row['customer_percent']:.1f}%)."
        )
        feature_summaries = [
            f"{feature} median {row[f'median_{feature}']:.2f}"
            for feature in features
        ]
        lines.append("  " + "; ".join(feature_summaries))
    lines.extend([
        "",
        "INTERPRETATION",
        "Segments are descriptive groups found in the uploaded data. They are not predictions, and unequal group sizes are a valid result.",
        "Silhouette and Davies-Bouldin are clustering diagnostics, not business success or prediction-accuracy scores. Review the profiles and business context.",
        "No profit, causal effect, or future purchase is claimed unless the uploaded data and a separate analysis support it.",
    ])
    return "\n".join(lines) + "\n"


st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5edf7; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] { color: #e5edf7; }
    [data-testid="stHeader"] { background: rgba(11,18,32,.95); }
    .hero { padding: 1.65rem 1.9rem; border-radius: 18px; background: linear-gradient(115deg,#10243a,#17606a); color: white; margin: .25rem 0 1.15rem; }
    .hero h1 { color: white !important; margin: 0; font-size: 2.15rem; }
    .hero p { color: #dce9ef !important; margin: .5rem 0 0; font-size: 1.02rem; }
    div[data-testid="stMetric"] { background: #121e30; border: 1px solid #25364b; padding: .9rem 1rem; border-radius: 14px; }
    div[data-testid="stFileUploader"] { background: #121e30; border: 1px solid #25364b; padding: 1rem; border-radius: 14px; }
    div.stButton > button[kind="primary"] { background: #147d80; border: 0; border-radius: 9px; padding: .65rem 1.2rem; }
    </style>
    <div class="hero"><h1>Customer Intelligence</h1>
    <p>Turn purchase data into clear customer groups and practical business insights.</p></div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Upload CSV or Excel • Analyse customer or transaction data • Export profiles and segment results"
)
st.info(
    "Privacy: uploaded files are processed by this app and are not written to project files. "
    "If hosted online, the file is sent to the hosting server for analysis; do not upload sensitive personal data.",
    icon="🔒",
)

uploaded_file = st.file_uploader(
    "Choose a customer or sales dataset",
    type=["csv", "xlsx", "xls"],
    help=f"Use one row per customer or transaction/item. Maximum file size: {MAX_FILE_MB} MB. Excel files use one worksheet at a time.",
)

if uploaded_file is None:
    st.markdown("### What you can do")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("**1 · Upload**  \nAdd a CSV or Excel file and inspect its data quality.")
    col_b.markdown("**2 · Prepare**  \nMap fields and aggregate transactions into customer profiles.")
    col_c.markdown("**3 · Understand**  \nExplore customer segments, edit their names, and download results.")
    st.stop()

if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
    st.error(f"This file is {uploaded_file.size / (1024 * 1024):.1f} MB. The current app limit is {MAX_FILE_MB} MB.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_digest = hashlib.sha256(file_bytes).hexdigest()
sheet_name = None
if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
    try:
        workbook = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_name = st.selectbox("Choose worksheet", workbook.sheet_names)
    except Exception as exc:
        st.error(f"Could not read this Excel workbook: {exc}")
        st.stop()

try:
    raw_data = load_uploaded_file(file_bytes, uploaded_file.name, sheet_name)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if raw_data.empty:
    st.error("This file has no data rows. Upload a non-empty CSV or Excel worksheet.")
    st.stop()
if len(raw_data.columns) < 2:
    st.warning("This file has fewer than two columns. Add the customer identifier and useful purchase measures.")

# Clean whitespace from headings and make duplicate heading labels unique.
raw_data.columns = [str(column).strip() or f"unnamed_{i + 1}" for i, column in enumerate(raw_data.columns)]
if raw_data.columns.duplicated().any():
    seen_headings: dict[str, int] = {}
    unique_headings = []
    for heading in raw_data.columns:
        seen_headings[heading] = seen_headings.get(heading, 0) + 1
        unique_headings.append(heading if seen_headings[heading] == 1 else f"{heading}_{seen_headings[heading]}")
    raw_data.columns = unique_headings
columns = list(raw_data.columns)
suggested = detect_mapping(columns)
initial_grain = guess_grain(raw_data, suggested)

tab_setup, tab_results = st.tabs(["1 · Prepare data", "2 · Analysis results"])

with tab_setup:
    st.subheader("Dataset overview")
    overview = st.columns(4)
    overview[0].metric("Rows", f"{len(raw_data):,}")
    overview[1].metric("Columns", f"{len(columns):,}")
    overview[2].metric("Duplicate rows", f"{int(raw_data.duplicated().sum()):,}")
    overview[3].metric("Missing cells", f"{int(raw_data.isna().sum().sum()):,}")
    st.dataframe(raw_data.head(12), width="stretch", hide_index=True)

    quality_table = pd.DataFrame({
        "Column": columns,
        "Detected type": [str(raw_data[c].dtype) for c in columns],
        "Missing values": [int(raw_data[c].isna().sum()) for c in columns],
        "Unique values": [int(raw_data[c].nunique(dropna=True)) for c in columns],
    })
    with st.expander("Review data quality by column"):
        st.dataframe(quality_table, width="stretch", hide_index=True)

    st.subheader("Tell us what each row represents")
    grain_options = [
        "Auto detect",
        "Customer-level (one row per customer)",
        "Transaction-level (one row per item/order)",
    ]
    grain_choice = st.radio(
        "Dataset grain",
        grain_options,
        index=0,
        horizontal=True,
        help="If a customer appears in many rows, use transaction-level so the app can aggregate their purchases.",
    )
    grain = initial_grain if grain_choice == "Auto detect" else grain_choice
    st.caption(f"Current assumption: **{grain}**. Change it if each row represents something else.")
    if int(raw_data.duplicated().sum()):
        st.warning(
            f"{int(raw_data.duplicated().sum()):,} exact duplicate rows were found. "
            "They are retained for review rather than silently removed; decide whether they are true duplicates or repeated purchases."
        )

    with st.expander("Map columns (check automatic suggestions)", expanded=True):
        st.write("Confirm the fields that describe customers and purchases. Leave unavailable fields unmapped.")
        st.caption(
            "For item-level files, map a per-line revenue field if available. If only unit price and quantity exist, "
            "the app multiplies them. Avoid mapping an order total repeated on every item row as line revenue."
        )
        customer_col = select_mapping("Customer ID", columns, suggested["customer_id"], "map_customer")
        if grain.startswith("Transaction-level"):
            order_col = select_mapping("Order / transaction ID", columns, suggested["order_id"], "map_order")
            product_col = select_mapping("Product name or ID", columns, suggested["product_id"], "map_product")
            category_col = select_mapping("Product category", columns, suggested["category"], "map_category")
            quantity_col = select_mapping("Quantity / units", columns, suggested["quantity"], "map_quantity")
            price_col = select_mapping("Unit price", columns, suggested["unit_price"], "map_price")
            revenue_col = select_mapping("Line revenue / sales amount", columns, suggested["revenue"], "map_revenue")
            date_col = select_mapping("Purchase date", columns, suggested["date"], "map_date")
            freight_col = select_mapping("Freight / shipping amount", columns, suggested["freight"], "map_freight")
        else:
            order_col = product_col = category_col = quantity_col = price_col = revenue_col = date_col = freight_col = None

    mapping = {
        "customer_id": customer_col,
        "order_id": order_col,
        "product_id": product_col,
        "category": category_col,
        "quantity": quantity_col,
        "unit_price": price_col,
        "revenue": revenue_col,
        "date": date_col,
        "freight": freight_col,
    }

    try:
        if grain.startswith("Transaction-level"):
            customer_data, preparation_quality, category_report = transaction_to_customers(raw_data, mapping)
            spend_definition = preparation_quality.get("spend_definition")
        else:
            customer_data, preparation_quality = customer_level_data(raw_data, customer_col)
            category_report = None
            spend_definition = None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Prepared customer profiles")
    st.write(
        f"The app prepared **{len(customer_data):,} customer rows**. It keeps the uploaded file unchanged and creates a separate working table for analysis."
    )
    if preparation_quality.get("rows_missing_customer_id", 0):
        st.warning(
            f"{preparation_quality['rows_missing_customer_id']:,} rows had no customer ID. "
            "Transaction rows without a customer ID were excluded from customer aggregation; customer-level rows are retained with generated labels."
        )
    if preparation_quality.get("duplicate_customer_ids", 0):
        st.warning(
            f"{preparation_quality['duplicate_customer_ids']:,} repeated customer IDs are present. "
            "If these rows represent separate purchases, choose transaction-level aggregation."
        )
    if preparation_quality.get("invalid_numeric_values_coerced_to_missing"):
        st.warning(
            "Some mapped numeric values could not be converted and are treated as missing. "
            "Review the selected columns and use consistent number formats."
        )
    if spend_definition:
        st.caption(f"Sales value will be calculated as: {spend_definition}. This is not profit.")
    if grain.startswith("Transaction-level") and not quantity_col and not revenue_col and not price_col:
        st.warning("No quantity, price, or revenue field was mapped. Customer spending cannot be calculated.")
    if grain.startswith("Transaction-level") and not order_col:
        st.info("No order ID was mapped. Order frequency and average order value cannot be calculated.")
    st.dataframe(customer_data.head(12), width="stretch", hide_index=True)
    with st.expander("Preparation checks"):
        st.json(preparation_quality)
    if category_report is not None and not category_report.empty:
        with st.expander("Category sales snapshot"):
            st.dataframe(category_report.head(20), width="stretch", hide_index=True)

    st.subheader("Choose customer measures")
    id_columns = {customer_col} if customer_col else set()
    id_columns.update(
        c for c in customer_data.columns
        if looks_like_identifier(c) and c not in {
            "distinct_orders", "distinct_products", "distinct_categories",
        }
    )
    candidates = numeric_candidates(customer_data, id_columns | {"customer_analysis_id"})
    if len(candidates) < 2:
        st.error(
            "At least two usable numeric customer measures are needed for K-means. "
            "Add measures such as spending, purchase frequency, quantity, or product counts."
        )
        st.stop()
    defaults = [
        c for c in candidates
        if any(hint in normalize_name(c) for hint in DEFAULT_FEATURE_HINTS)
    ]
    if len(defaults) < 2:
        defaults = candidates[: min(5, len(candidates))]
    selected_features = st.multiselect(
        "Features for customer segmentation",
        options=candidates,
        default=defaults,
        help="Customer IDs and order IDs are excluded. Use measures that describe behavior, not identifiers.",
    )
    if len(selected_features) < 2:
        st.info("Select at least two features to continue.")
        st.stop()

    feature_frame = pd.DataFrame(index=customer_data.index)
    invalid_counts = {}
    for feature in selected_features:
        feature_frame[feature], invalid_counts[feature] = numeric_series(customer_data[feature])
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    all_missing = [c for c in feature_frame if feature_frame[c].notna().sum() == 0]
    constants = [c for c in feature_frame if feature_frame[c].nunique(dropna=True) <= 1]
    if all_missing or constants:
        removed = sorted(set(all_missing + constants))
        st.warning(f"These selected columns cannot help separate customers and will be skipped: {', '.join(removed)}.")
        feature_frame = feature_frame.drop(columns=removed)
    if feature_frame.shape[1] < 2:
        st.error("After removing unusable columns, fewer than two numeric features remain.")
        st.stop()
    missing_features = feature_frame.isna().sum()
    for feature in feature_frame:
        median = feature_frame[feature].median()
        feature_frame[feature] = feature_frame[feature].fillna(median)
    if int(missing_features.sum()):
        st.info(
            f"{int(missing_features.sum()):,} missing or invalid feature values will be filled with each feature's median. "
            "No customer rows are removed for missing feature values."
        )

    log_skewed = st.checkbox(
        "Reduce extreme right-skew in non-negative features before clustering",
        value=True,
        help="Uses log(1 + value) only for selected non-negative features with strong right skew. Profiles remain in original units.",
    )
    model_features = feature_frame.astype("float64").copy()
    transformed_features = []
    if log_skewed:
        for feature in model_features:
            skew = model_features[feature].skew()
            if model_features[feature].min() >= 0 and pd.notna(skew) and skew > 1.5:
                model_features[feature] = np.log1p(model_features[feature])
                transformed_features.append(feature)

    # Exclude rows with duplicate customer IDs only in the customer-level output is a user choice;
    # K-means treats each retained row as a record and the warning above makes this visible.
    unique_profiles = int(model_features.drop_duplicates().shape[0])
    max_k = min(8, len(model_features) - 1, unique_profiles)
    if max_k < 2:
        st.error("This dataset does not contain enough distinct customer profiles to form two clusters.")
        st.stop()
    k_options = list(range(2, max_k + 1))
    k = st.select_slider(
        "Number of customer groups",
        options=k_options,
        value=min(3, max_k),
        help="Choose a value, then review the model diagnostics and whether the profiles are useful for your business.",
    )
    if len(model_features) > MAX_TRAIN_ROWS:
        st.caption(
            f"For speed, the model will fit on a deterministic sample of {MAX_TRAIN_ROWS:,} customers "
            f"and assign all {len(model_features):,} customers to the resulting groups."
        )
    if transformed_features:
        st.caption("Log-transformed for clustering: " + ", ".join(transformed_features))

    analysis_signature = hashlib.sha256(
        (
            file_digest
            + repr((sheet_name, grain, mapping, selected_features, k, log_skewed))
        ).encode("utf-8")
    ).hexdigest()
    results_state_key = "customer_intelligence_result"

    if st.button("Run customer analysis", type="primary", width="stretch"):
        try:
            with st.spinner("Preparing features, fitting K-means, and building customer profiles…"):
                scaler = StandardScaler()
                scaled = scaler.fit_transform(model_features.to_numpy(dtype=np.float64))
                if len(scaled) > MAX_TRAIN_ROWS:
                    rng = np.random.default_rng(RANDOM_STATE)
                    train_indices = np.sort(rng.choice(len(scaled), MAX_TRAIN_ROWS, replace=False))
                    training_scaled = scaled[train_indices]
                else:
                    training_scaled = scaled
                model = KMeans(
                    n_clusters=int(k),
                    random_state=RANDOM_STATE,
                    n_init=10,
                    max_iter=300,
                )
                training_labels = model.fit_predict(training_scaled)
                if len(np.unique(training_labels)) < 2:
                    raise ValueError("K-means created fewer than two groups. Try another group count or different features.")
                labels = model.predict(scaled)
                score_n = min(SILHOUETTE_SAMPLE, len(training_scaled))
                if score_n < len(training_scaled):
                    score = silhouette_score(
                        training_scaled,
                        training_labels,
                        sample_size=score_n,
                        random_state=RANDOM_STATE,
                    )
                    eval_labels = training_labels
                    eval_scaled = training_scaled
                else:
                    eval_labels = training_labels
                    eval_scaled = training_scaled
                    score = silhouette_score(eval_scaled, eval_labels)
                dbi = davies_bouldin_score(eval_scaled, eval_labels)

                assigned = customer_data.copy()
                assigned["cluster_id"] = labels
                suggested_names = suggest_names(assigned, list(feature_frame.columns))
                assigned["segment_name"] = assigned["cluster_id"].map(suggested_names)
                profile = feature_frame.copy()
                profile["cluster_id"] = labels
                profile_table = profile.groupby("cluster_id").agg(
                    customer_count=(feature_frame.columns[0], "size"),
                    **{f"mean_{c}": (c, "mean") for c in feature_frame.columns},
                    **{f"median_{c}": (c, "median") for c in feature_frame.columns},
                )
                profile_table["customer_percent"] = (
                    profile_table["customer_count"] / len(assigned) * 100
                )
                profile_table = profile_table.reset_index()
                profile_table["segment_name"] = profile_table["cluster_id"].map(suggested_names)
                summary = build_summary(
                    uploaded_file.name,
                    grain,
                    assigned,
                    profile_table,
                    list(feature_frame.columns),
                    int(k),
                    float(score),
                    float(dbi),
                    str(spend_definition) if spend_definition else None,
                    len(training_scaled),
                )
                st.session_state[results_state_key] = {
                    "signature": analysis_signature,
                    "customers": assigned,
                    "profiles": profile_table,
                    "features": list(feature_frame.columns),
                    "names": suggested_names,
                    "k": int(k),
                    "silhouette": float(score),
                    "davies_bouldin": float(dbi),
                    "inertia": float(model.inertia_),
                    "training_rows": len(training_scaled),
                    "transformed": transformed_features,
                    "summary": summary,
                    "spend_definition": spend_definition,
                    "category_report": category_report,
                    "quality": preparation_quality,
                }
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")

    completed_state = st.session_state.get(results_state_key)
    if completed_state and completed_state.get("signature") == analysis_signature:
        st.success(
            "Analysis is ready. Open the **2 · Analysis results** tab above to view the dashboard and download files."
        )

with tab_results:
    state = st.session_state.get("customer_intelligence_result")
    if not state or state.get("signature") != analysis_signature:
        st.info("Finish the setup on the **Prepare data** tab, then run customer analysis to see results.")
    else:
        customers = state["customers"]
        features = state["features"]
        profiles = state["profiles"].copy()
        names = dict(state["names"])
        st.subheader("Business overview")
        metrics = st.columns(4)
        metrics[0].metric("Customers analysed", f"{len(customers):,}")
        metrics[1].metric("Customer groups", str(state["k"]))
        spend_col = find_alias(list(customers.columns), [
            "total_spend", "total_item_spend", "total_revenue", "revenue",
            "sales", "amount_spent", "spend", "monetary_value",
        ])
        order_col = find_alias(list(customers.columns), [
            "distinct_orders", "order_count", "purchase_frequency", "orders",
        ])
        if spend_col and pd.to_numeric(customers[spend_col], errors="coerce").notna().any():
            metrics[2].metric("Average customer spend", f"{pd.to_numeric(customers[spend_col], errors='coerce').mean():,.2f}")
        else:
            metrics[2].metric("Spend metric", "Not available")
        if order_col:
            repeat_share = (pd.to_numeric(customers[order_col], errors="coerce") > 1).mean() * 100
            metrics[3].metric("Repeat customers", f"{repeat_share:.1f}%")
        else:
            metrics[3].metric("Order frequency", "Not available")

        if state.get("spend_definition"):
            st.caption(f"Sales definition: {state['spend_definition']}. Sales value is not profit.")
        st.subheader("Suggested segment names")
        st.caption(
            "Suggestions are generated from each group's measured profile. Rename them for your business. "
            "Group sizes are based on the data and are not forced to be equal."
        )
        name_columns = st.columns(min(3, len(names)))
        for index, cluster in enumerate(sorted(names)):
            with name_columns[index % len(name_columns)]:
                names[cluster] = st.text_input(
                    f"Group {cluster + 1}",
                    value=names[cluster],
                    key=f"name_{state['signature'][:10]}_{cluster}",
                ).strip() or f"Group {cluster + 1}"
        customers["segment_name"] = customers["cluster_id"].map(names)
        profiles["segment_name"] = profiles["cluster_id"].map(names)

        st.subheader("Group size and purchase behavior")
        profile_columns = ["cluster_id", "segment_name", "customer_count", "customer_percent"]
        profile_view = profiles[profile_columns].copy()
        profile_view["customer_percent"] = profile_view["customer_percent"].round(1)
        st.dataframe(profile_view, width="stretch", hide_index=True)
        chart_data = profile_view.set_index("segment_name")["customer_count"]
        st.bar_chart(chart_data, y_label="Customers")
        st.caption(
            "Unequal group sizes often reflect real customer behavior. K-means does not promise equal-sized groups."
        )

        st.subheader("Customer profile by group")
        display_profile = profiles[["segment_name", *[
            item for feature in features
            for item in (f"median_{feature}", f"mean_{feature}")
        ]]].copy()
        st.dataframe(display_profile.round(2), width="stretch", hide_index=True)

        category_report = state.get("category_report")
        if category_report is not None and not category_report.empty:
            st.subheader("Top product categories")
            if state.get("quality", {}).get("categories_inferred_from_description"):
                st.caption(
                    "Categories are broad estimates inferred from product descriptions because this dataset has no official category field. "
                    "Review the labels before using them for business decisions."
                )
            st.dataframe(category_report.head(10), width="stretch", hide_index=True)

        st.subheader("What each customer group tends to buy")
        if not any(column in customers.columns for column in ("age", "customer_age", "gender", "customer_gender", "income", "household_income")):
            st.caption(
                "These groups describe purchase behaviour. This dataset does not include age, gender, or income, "
                "so the app does not infer demographic groups."
            )
        favorite_category_col = next(
            (column for column in ("most_purchased_category", "favorite_category", "preferred_category")
             if column in customers.columns),
            None,
        )
        if favorite_category_col:
            preference = customers[["segment_name", favorite_category_col]].dropna()
            preference = preference.rename(columns={favorite_category_col: "Preferred category"})
            preference = preference.groupby(["segment_name", "Preferred category"]).size().rename("Customers").reset_index()
            top_preferences = preference.sort_values("Customers", ascending=False)
            st.caption(
                "Each customer is counted once under their most frequently purchased category. "
                "This is customer preference, not category revenue or units sold."
            )
            st.dataframe(top_preferences.head(20), width="stretch", hide_index=True)
            preference_chart = top_preferences.pivot(
                index="segment_name", columns="Preferred category", values="Customers"
            ).fillna(0)
            chart_categories = top_preferences.groupby("Preferred category")["Customers"].sum().nlargest(8).index
            st.bar_chart(preference_chart.reindex(columns=chart_categories, fill_value=0), height=360)
        else:
            st.info(
                "No product category or usable item description was available. Upload transaction rows and map a category or product description column."
            )

        price_col = next(
            (column for column in ("average_unit_price", "avg_unit_price", "unit_price", "average_price")
             if column in customers.columns),
            None,
        )
        if price_col:
            prices = pd.to_numeric(customers[price_col], errors="coerce")
            valid_prices = prices.dropna()
            if not valid_prices.empty and valid_prices.nunique() > 1:
                premium_cutoff = float(valid_prices.quantile(0.75))
                premium_mask = prices >= premium_cutoff
                premium_count = int(premium_mask.sum())
                premium_share = premium_count / int(prices.notna().sum()) * 100
                st.subheader("Price-point behaviour")
                p1, p2, p3 = st.columns(3)
                p1.metric("Median item price", f"{valid_prices.median():,.2f}")
                p2.metric("Higher-price shoppers", f"{premium_count:,}")
                p3.metric("Share of customers", f"{premium_share:.1f}%")
                st.caption(
                    f"Higher-price shoppers are customers whose average item price is at or above this dataset's 75th percentile ({premium_cutoff:,.2f}). "
                    "This is a relative price tier within the uploaded data, not a universal luxury threshold."
                )
                price_profile = customers.assign(_item_price=prices).groupby("segment_name")["_item_price"].agg(
                    median_item_price="median", higher_price_customers=lambda values: int((values >= premium_cutoff).sum())
                ).reset_index()
                st.dataframe(price_profile.round(2), width="stretch", hide_index=True)
        else:
            st.info(
                "For premium-price shopper analysis, upload transaction rows and map a unit price column. "
                "The app will compare each customer's average item price with the uploaded dataset's upper quartile."
            )

        with st.expander("How to read model diagnostics"):
            diag1, diag2, diag3 = st.columns(3)
            diag1.metric("Silhouette", f"{state['silhouette']:.4f}")
            diag2.metric("Davies–Bouldin", f"{state['davies_bouldin']:.4f}")
            diag3.metric("Inertia", f"{state['inertia']:,.2f}")
            st.write(
                "Silhouette and Davies–Bouldin describe separation and compactness in the selected feature space. "
                "They are not prediction accuracy or proof that the groups are useful to a business. "
                "Interpret the profiles, feature choices, and context together."
            )
            st.write(
                f"The scaler and K-means model were fitted on {state['training_rows']:,} customer rows; "
                "the model assigned all analysed customers to a group."
            )

        st.subheader("Customer assignments")
        st.dataframe(customers.head(100), width="stretch", hide_index=True)

        # Rebuild the summary after user-edited names so every download matches this screen.
        profiles["segment_name"] = profiles["cluster_id"].map(names)
        summary_text = build_summary(
            uploaded_file.name,
            grain,
            customers,
            profiles,
            features,
            state["k"],
            state["silhouette"],
            state["davies_bouldin"],
            str(state["spend_definition"]) if state.get("spend_definition") else None,
            state["training_rows"],
        )
        st.subheader("Download your results")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "Customer profiles CSV",
            data=customer_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_level_data.csv",
            mime="text/csv",
            width="stretch",
        )
        d2.download_button(
            "Segment assignments CSV",
            data=customers.to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_segment_assignments.csv",
            mime="text/csv",
            width="stretch",
        )
        d3.download_button(
            "Cluster profiles CSV",
            data=profiles.to_csv(index=False).encode("utf-8-sig"),
            file_name="cluster_profiles.csv",
            mime="text/csv",
            width="stretch",
        )
        d4.download_button(
            "Analysis summary TXT",
            data=summary_text.encode("utf-8"),
            file_name="analysis_summary.txt",
            mime="text/plain",
            width="stretch",
        )
