# Maxima Wealth Dashboard – MVP (Streamlit Cloud)

A web-ready MVP dashboard for Forex / Gold / Index / Stocks / Crypto mock data.

## Files
- `app.py` — Streamlit app (sidebar filters, tabs, downloads)
- `requirements.txt` — Python dependencies
- `Clean_MockData.xlsx` — sample mock dataset (sheet name: `MockData`)
- `.streamlit/config.toml` — optional theme

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://streamlit.io/cloud → **New app** → connect GitHub.
3. Select your repo / branch; set **Main file path** to `app.py`.
4. Click **Deploy**. You’ll get a public URL.
5. (Optional) If you prefer not to ship sample data, delete `Clean_MockData.xlsx`. The app lets users upload their own Excel in the sidebar.

## Notes
- Default sheet name is `MockData`. You can change it in the sidebar.
- If you need PNG chart downloads, `kaleido` is already listed in `requirements.txt`.
