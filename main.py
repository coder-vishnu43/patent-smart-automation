import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import random

# -----------------------------
# Helper functions
# -----------------------------
def clean_dataframe(df):
    # Standard columns
    cols_map = {
        'Publication Title': 'Title',
        'Patent Title': 'Title',
        'Title': 'Title',
        'Publication Year': 'Year',
        'Year': 'Year',
        'Applicants': 'Organization',
        'Assignee': 'Organization',
        'Organization': 'Organization',
        'Technology Field': 'Domain',
        'Domain': 'Domain',
        'Field': 'Domain',
        'Abstract': 'Keywords',
        'Keywords': 'Keywords',
        'TRL': 'TRL'
    }

    df_clean = pd.DataFrame()
    for original, new in cols_map.items():
        if original in df.columns:
            df_clean[new] = df[original]

    # Fill missing
    df_clean['Title'] = df_clean.get('Title', pd.Series(["Unknown Title"] * len(df)))
    df_clean['Organization'] = df_clean.get('Organization', pd.Series(["Unknown Org"] * len(df)))
    df_clean['Keywords'] = df_clean.get('Keywords', pd.Series(["N/A"] * len(df)))

    # Year column
    if 'Year' not in df_clean.columns:
        df_clean['Year'] = "Unknown"
    else:
        df_clean['Year'] = df_clean['Year'].fillna("Unknown").astype(str)

    # TRL column
    if 'TRL' not in df_clean.columns:
        df_clean['TRL'] = [random.randint(3, 9) for _ in range(len(df_clean))]

    # AI-style Domain detection
    def detect_domain(row):
        text = str(row['Title']) + " " + str(row['Keywords'])
        text = text.lower()
        if any(k in text for k in ["quantum", "qubit", "qkd"]):
            return "Quantum"
        elif any(k in text for k in ["robot", "drone", "autonomous"]):
            return "Robotics"
        elif any(k in text for k in ["ai", "ml", "neural", "deep learning"]):
            return "AI"
        elif any(k in text for k in ["cyber", "encryption", "security"]):
            return "Cybersecurity"
        elif any(k in text for k in ["bio", "genome", "dna"]):
            return "Biotech"
        else:
            return "Other"

    if 'Domain' not in df_clean.columns or df_clean['Domain'].isna().all():
        df_clean['Domain'] = df_clean.apply(detect_domain, axis=1)

    return df_clean[['Title', 'Year', 'Organization', 'Keywords', 'Domain', 'TRL']]

def generate_alerts(df):
    alerts = []
    try:
        latest_year = df['Year'].replace("Unknown", np.nan).dropna().astype(int).max()
        alerts.append(f"📈 Surge of patents detected in **{latest_year}**.")
    except:
        pass

    if not df['Domain'].empty:
        top_domain = df['Domain'].mode()[0]
        alerts.append(f"🔍 Most active domain: **{top_domain}**")

    if not df['Organization'].empty:
        top_org = df['Organization'].mode()[0]
        alerts.append(f"🏢 Leading organization: **{top_org}**")

    if not df['TRL'].empty and df['TRL'].mean() > 6:
        alerts.append("⚡ Overall TRL levels indicate advanced technology readiness.")

    return alerts

def compute_s_curve(df):
    df_year = df.groupby('Year').size().reset_index(name='Count')
    df_year['Cumulative'] = df_year['Count'].cumsum()
    df_year['S_curve'] = df_year['Cumulative'] / df_year['Cumulative'].max()

    # Trend Forecast (Linear Regression)
    try:
        df_year_num = df_year[df_year['Year'] != "Unknown"].copy()
        df_year_num['Year_int'] = df_year_num['Year'].astype(int)
        X = df_year_num['Year_int'].values.reshape(-1,1)
        y = df_year_num['Count'].values
        model = LinearRegression().fit(X, y)
        next_year = np.array([[X.max()+1]])
        pred = int(model.predict(next_year)[0])
        df_year = df_year.append({'Year': str(int(X.max()+1)), 'Count': pred,
                                  'Cumulative': df_year['Cumulative'].max()+pred,
                                  'S_curve': (df_year['Cumulative'].max()+pred)/df_year['Cumulative'].max()},
                                 ignore_index=True)
    except:
        pass

    return df_year

def domain_convergence(df):
    domains = df['Domain'].unique()
    convergence = {}
    for i, d1 in enumerate(domains):
        for d2 in domains[i+1:]:
            k1 = df[df['Domain']==d1]['Keywords'].apply(lambda x: set(str(x).lower().split()))
            k2 = df[df['Domain']==d2]['Keywords'].apply(lambda x: set(str(x).lower().split()))
            overlap = sum(len(a & b) for a,b in zip(k1,k2))
            convergence[f"{d1} + {d2}"] = overlap
    return convergence

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title=" Smart Tech Intelligence", layout="wide")
st.title("🔬 Smart Technology Intelligence Dashboard")

# Sidebar
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Upload CSV or use default dataset:")

uploaded_file = st.sidebar.file_uploader("📂 Upload Patent CSV", type=["csv"])

# Load Data
if uploaded_file is None:
    st.info("No file uploaded. Showing default dataset.")
    try:
        df = pd.read_csv("data/multi_domain_demo_data_clean.csv")  # keep default CSV
        df = clean_dataframe(df)
    except:
        st.error("Default CSV not found. Please upload a CSV.")
        st.stop()
else:
    raw_df = pd.read_csv(uploaded_file)
    df = clean_dataframe(raw_df)
    st.success("✅ CSV uploaded & cleaned successfully!")

# Filters
year_options = sorted(df['Year'].astype(str).unique())
year_filter = st.sidebar.multiselect("Select Year(s):", options=year_options, default=year_options)

domain_options = sorted(df['Domain'].unique())
domain_filter = st.sidebar.multiselect("Select Domain(s):", options=domain_options, default=domain_options)

df_filtered = df[df['Year'].astype(str).isin(year_filter) & df['Domain'].isin(domain_filter)]

# Download button
st.sidebar.markdown("---")
st.sidebar.markdown("💾 Download Processed Data")
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_patents.csv',
    mime='text/csv'
)

# -----------------------------
# Dataset Table
# -----------------------------
st.subheader("📊 Patent Dataset")
st.dataframe(df_filtered, use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df_filtered, x="Year", color="Domain", title="Patents Over Time")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    domain_count = df_filtered['Domain'].value_counts().reset_index()
    domain_count.columns = ['Domain', 'Count']
    fig2 = px.bar(domain_count, x='Domain', y='Count', title="Patents by Domain")
    st.plotly_chart(fig2, use_container_width=True)

org_count = df_filtered['Organization'].value_counts().reset_index()
org_count.columns = ['Organization', 'Count']
fig3 = px.bar(org_count.head(10), x='Organization', y='Count', title="Top Organizations")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.box(df_filtered, x="Domain", y="TRL", title="TRL Distribution by Domain")
st.plotly_chart(fig4, use_container_width=True)

# S-curve
st.subheader("📈 Technology Trend Analysis")
s_curve_df = compute_s_curve(df_filtered)
fig5 = px.line(s_curve_df, x='Year', y='S_curve', markers=True, title="S-Curve of Patent Growth")
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.bar(s_curve_df, x='Year', y='Count', title="Hype Curve: Patents per Year", color='Count', color_continuous_scale='Viridis')
st.plotly_chart(fig6, use_container_width=True)

# Domain convergence
st.subheader("🔗 Domain Convergence (Keyword Overlap)")
convergence = domain_convergence(df_filtered)
if convergence:
    df_conv = pd.DataFrame(list(convergence.items()), columns=['Domain Pair', 'Keyword Overlap Count'])
    st.dataframe(df_conv.sort_values(by='Keyword Overlap Count', ascending=False))
else:
    st.info("No convergence detected.")

# Alerts / Insights
st.subheader("🚨 AI-style Insights & Alerts")
alerts = generate_alerts(df_filtered)
for a in alerts:
    st.warning(a)
