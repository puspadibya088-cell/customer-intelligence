# Customer Intelligence

A reusable Streamlit app for customer segmentation and purchase behaviour analysis. Upload CSV, XLSX, or XLS files with customer-level data or transaction/item-level rows.

## Run locally

Install Python 3.11 or 3.12, open a terminal in this folder, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Create a **new public GitHub repository containing only the files in this folder**. Do not upload the larger project package or any customer uploads.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Choose the repository, branch `main`, and entry file `app.py`.
4. Pick an available app address such as `your-name-customer-intelligence.streamlit.app` and deploy.

The Streamlit-provided `*.streamlit.app` URL is free. A custom domain you own (for example `.com`) is separate and may cost money. You must sign in and complete any provider account/terms steps yourself.

## Insights and data limits

For transaction data, map customer ID, product category, unit price, quantity, line revenue, and order ID where present. The app aggregates transactions into customer profiles, compares category preferences across segments, and marks higher-price shoppers relative to the uploaded data's 75th percentile of average item prices. Category preference counts each customer once by their most frequent category.

Groups are based on purchase measures, not demographics. Age, gender, or income conclusions are only possible when those fields are provided and suitable for analysis. Do not upload sensitive customer data to a public app; uploaded files are processed on the hosting server.

K-means group sizes are not forced to be equal. Silhouette and Davies-Bouldin are diagnostics, not proof of business value or predictive accuracy.
