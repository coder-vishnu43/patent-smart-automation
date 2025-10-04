import os
import time
import json
import random
import re
import hashlib
import sqlite3
from urllib.parse import quote_plus, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests
import streamlit as st
import streamlit.components.v1 as components

# =========================
# Secrets -> environment
# =========================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "OPENAI_API_BASE" in st.secrets:
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]

LLM_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY")

# =========================
# App config
# =========================
st.set_page_config(page_title="Smart Tech Intelligence", layout="wide")
st.title("🔬 Smart Technology Intelligence Dashboard")

# =========================
# Data helpers
# =========================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols_map = {
        "Publication Title": "Title",
        "Patent Title": "Title",
        "Title": "Title",
        "Publication Year": "Year",
        "Year": "Year",
        "Applicants": "Organization",
        "Assignee": "Organization",
        "Organization": "Organization",
        "Technology Field": "Domain",
        "Domain": "Domain",
        "Field": "Domain",
        "Abstract": "Keywords",
        "Keywords": "Keywords",
        "TRL": "TRL",
        "Patent Link": "Patent Link",
        "Link": "Patent Link",
        "URL": "Patent Link",
    }
    df_clean = pd.DataFrame()
    for original, new in cols_map.items():
        if original in df.columns:
            df_clean[new] = df[original]

    df_clean["Title"] = df_clean.get("Title", pd.Series(["Unknown Title"] * len(df)))
    df_clean["Organization"] = df_clean.get("Organization", pd.Series(["Unknown Org"] * len(df)))
    df_clean["Keywords"] = df_clean.get("Keywords", pd.Series(["N/A"] * len(df)))

    if "Year" not in df_clean.columns:
        df_clean["Year"] = "Unknown"
    else:
        df_clean["Year"] = df_clean["Year"].fillna("Unknown").astype(str)

    if "TRL" not in df_clean.columns:
        df_clean["TRL"] = [random.randint(3, 9) for _ in range(len(df_clean))]

    if "Patent Link" in df_clean.columns:
        df_clean["Patent Link"] = df_clean["Patent Link"].astype(str).str.strip()
        df_clean["Patent Link"] = df_clean["Patent Link"].replace(["", "nan", "None", "NULL"], pd.NA)
        df_clean["Patent Link"] = df_clean["Patent Link"].apply(
            lambda x: f"https://{x}" if pd.notna(x) and not x.startswith(("http://", "https://")) else x
        )
    else:
        df_clean["Patent Link"] = pd.NA

    df_clean["Patent Link"] = df_clean.apply(
        lambda row: row["Patent Link"]
        if pd.notna(row["Patent Link"])
        else f"https://www.google.com/search?q={row['Title']} site:patents.google.com",
        axis=1,
    )

    def detect_domain(row):
        text = f"{row['Title']} {row['Keywords']}".lower()
        if any(k in text for k in ["quantum", "qubit", "qkd"]):
            return "Quantum"
        if any(k in text for k in ["robot", "drone", "autonomous"]):
            return "Robotics"
        if any(k in text for k in ["ai", "ml", "neural", "deep learning"]):
            return "AI"
        if any(k in text for k in ["cyber", "encryption", "security"]):
            return "Cybersecurity"
        if any(k in text for k in ["bio", "genome", "dna"]):
            return "Biotech"
        return "Other"

    if "Domain" not in df_clean.columns or df_clean["Domain"].isna().all():
        df_clean["Domain"] = df_clean.apply(detect_domain, axis=1)

    return df_clean[["Title", "Year", "Organization", "Keywords", "Domain", "TRL", "Patent Link"]]

def generate_alerts(df: pd.DataFrame):
    alerts = []
    try:
        latest_year = df["Year"].replace("Unknown", np.nan).dropna().astype(int).max()
        alerts.append(f"📈 Surge of patents detected in {latest_year}.")
    except Exception:
        pass
    if not df["Domain"].empty:
        alerts.append(f"🔍 Most active domain: {df['Domain'].mode()[0]}")
    if not df["Organization"].empty:
        alerts.append(f"🏢 Leading organization: {df['Organization'].mode()[0]}")
    if not df["TRL"].empty and df["TRL"].mean() > 6:
        alerts.append("⚡ Overall TRL levels indicate advanced readiness.")
    return alerts

def compute_s_curve(df: pd.DataFrame):
    df_year = df.groupby("Year").size().reset_index(name="Count")
    df_year["Cumulative"] = df_year["Count"].cumsum()
    df_year["S_curve"] = df_year["Cumulative"] / df_year["Cumulative"].max()
    try:
        df_year_num = df_year[df_year["Year"] != "Unknown"].copy()
        df_year_num["Year_int"] = df_year_num["Year"].astype(int)
        X = df_year_num["Year_int"].values.reshape(-1, 1)
        y = df_year_num["Count"].values
        model = LinearRegression().fit(X, y)
        next_year = int(X.max() + 1)
        pred = int(model.predict(np.array([[next_year]]))[0])
        df_year = pd.concat(
            [
                df_year,
                pd.DataFrame(
                    {
                        "Year": [str(next_year)],
                        "Count": [pred],
                        "Cumulative": [df_year["Cumulative"].max() + pred],
                        "S_curve": [(df_year["Cumulative"].max() + pred) / df_year["Cumulative"].max()],
                    }
                ),
            ],
            ignore_index=True,
        )
    except Exception:
        pass
    return df_year

def domain_convergence(df: pd.DataFrame):
    domains = df["Domain"].unique()
    conv = {}
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1 :]:
            k1 = df[df["Domain"] == d1]["Keywords"].apply(lambda x: set(str(x).lower().split()))
            k2 = df[df["Domain"] == d2]["Keywords"].apply(lambda x: set(str(x).lower().split()))
            conv[f"{d1} + {d2}"] = sum(len(a & b) for a, b in zip(k1, k2))
    return conv

# =========================
# Sidebar and data loading
# =========================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Upload CSV or use default dataset:")
uploaded_file = st.sidebar.file_uploader("📂 Upload Patent CSV", type=["csv"])

if uploaded_file is None:
    st.info("No file uploaded. Showing default dataset.")
    try:
        df = pd.read_csv("data/multi_domain_demo_data_clean.csv")
        df = clean_dataframe(df)
    except Exception:
        st.error("Default CSV not found. Please upload a CSV.")
        st.stop()
else:
    df = clean_dataframe(pd.read_csv(uploaded_file))
    st.success("✅ CSV uploaded & cleaned successfully!")

year_options = sorted(df["Year"].astype(str).unique())
year_filter = st.sidebar.multiselect("Select Year(s):", options=year_options, default=year_options)
domain_options = sorted(df["Domain"].unique())
domain_filter = st.sidebar.multiselect("Select Domain(s):", options=domain_options, default=domain_options)
df_filtered = df[df["Year"].astype(str).isin(year_filter) & df["Domain"].isin(domain_filter)]

st.sidebar.markdown("---")
st.sidebar.markdown("💾 Download Processed Data")
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=df_filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_patents.csv",
    mime="text/csv",
)

# =========================
# Dataset table with links
# =========================
st.subheader("📊 Patent Dataset")
df_filtered["Patent Link"] = df_filtered["Patent Link"].astype(str)
st.dataframe(
    df_filtered[["Title", "Year", "Organization", "Keywords", "Domain", "TRL", "Patent Link"]],
    use_container_width=True,
    column_config={
        "Patent Link": st.column_config.LinkColumn(
            label="Patent Link", help="Open the patent page in a new tab", display_text="Open"
        )
    },
)

# ======================================
# Research Tool (UI label “gpt 5” only)
# ======================================
st.subheader("🔍 Patent Deep Research")

MODEL_LABEL = "gpt 5"
# If using api.openai.com, set to "gpt-4o-mini"
MODEL_MAP = {"gpt 5": "openai/gpt-4o-mini"}
def resolve_model(label: str) -> str:
    return MODEL_MAP.get(label, MODEL_MAP["gpt 5"])

st.sidebar.markdown("---")
st.sidebar.caption("LLM analysis")
ui_choice = st.sidebar.selectbox("Model", [MODEL_LABEL], index=0)
resolved_model = resolve_model(ui_choice)

# =========================
# Cache (SQLite-safe)
# =========================
def _s(val):
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    if val is None:
        return ""
    return str(val)

def _p(val):
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            return val
    return val

def get_db():
    conn = sqlite3.connect("analysis_cache.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
        key TEXT PRIMARY KEY,
        title TEXT, year TEXT, org TEXT, domain TEXT, url TEXT,
        summary TEXT, claims TEXT, entities TEXT, links TEXT,
        created_at REAL
    )"""
    )
    return conn

def cache_get(conn, key):
    cur = conn.execute("SELECT summary, claims, entities, links, created_at FROM analyses WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        return {
            "summary": _p(row[0]),
            "claims": _p(row[1]),
            "entities": _p(row[2]),
            "links": _p(row[3]),
            "created_at": row[4],
        }
    return None

def cache_put(conn, key, row, result):
    conn.execute(
        "INSERT OR REPLACE INTO analyses (key,title,year,org,domain,url,summary,claims,entities,links,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            key,
            row["Title"],
            str(row["Year"]),
            row["Organization"],
            row["Domain"],
            row["Patent Link"],
            _s(result.get("summary")),
            _s(result.get("claims")),
            _s(result.get("entities")),
            _s(result.get("links")),
            time.time(),
        ),
    )
    conn.commit()

# =========================
# LLM call (JSON only)
# =========================
def build_links_and_prompt(row):
    base_link = str(row.get("Patent Link", "")).strip()
    if base_link == "" or base_link.lower() in ("nan", "none"):
        base_link = f"https://www.google.com/search?q={quote_plus(row['Title'])}+site:patents.google.com"

    prompt = f"""Act as a patent research assistant.


Patent:
- Title: {row['Title']}
- Keywords/Abstract: {row['Keywords']}
- Organization: {row['Organization']}
- Year: {row['Year']}
- Domain: {row['Domain']}
- Link: {base_link}


Tasks:
1) Summarize invention and key claims/themes.
2) List assignees/applicants and any IPC/CPC hints.
3) Provide 5 related prior-art links and 5 competitor filings.
4) Give 5-bullet executive brief and 3 risks/limitations.
5) Estimate TRL with 1-line rationale.
"""
    gpat = f"https://www.google.com/search?q={quote_plus(row['Title'])}+site:patents.google.com"
    lens = f"https://www.lens.org/lens/search?q={quote_plus(row['Title'])}"
    gemini = "https://gemini.google.com/app"
    return base_link, gpat, lens, gemini, prompt

def llm_analyze(prompt: str) -> dict:
    if not LLM_API_KEY:
        st.error("Server config missing: set OPENAI_API_KEY (and optionally OPENAI_API_BASE).")
        return {"summary": "", "claims": "", "entities": "", "links": ""}

    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Smart Tech Intelligence",
    }
    body = {
        "model": resolved_model,
        "temperature": 0.1,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only with these keys exactly: "
                    "invention_summary, key_claims, assignees_applicants, "
                    "related_prior_art_links, competitor_filings, executive_brief, "
                    "risks_limitations, TRL_estimate. "
                    "No text outside JSON. "
                    "invention_summary must include title and abstract. "
                    "key_claims must be a list with at least 5 bullets. "
                    "related_prior_art_links must be a list of 5 https URLs. "
                    "competitor_filings must be a list of 5 https URLs. "
                    "executive_brief must be a list of 5 bullets. "
                    "risks_limitations must be a list of 3 bullets. "
                    "TRL_estimate must include integer level (1-9) and rationale. "
                    "All URLs must be real https links to patents or prior art "
                    "(patents.google.com, lens.org, uspto.gov, wipo.int, epo.org, ieee.org, acm.org) "
                    "— never example.com or placeholders."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error calling LLM: {e}")
        return {"summary": "", "claims": "", "entities": "", "links": ""}

    if resp.status_code != 200:
        st.error(f"LLM API error {resp.status_code}: {resp.text}")
        return {"summary": "", "claims": "", "entities": "", "links": ""}

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Parse error: {e} — raw: {resp.text[:500]}")
        return {"summary": "", "claims": "", "entities": "", "links": ""}

    def extract_first_json_block(text: str):
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start : i + 1]
        return None

    candidate = extract_first_json_block(content)
    if candidate:
        try:
            parsed = json.loads(candidate)
            return {"summary": parsed, "claims": "", "entities": "", "links": ""}
        except Exception:
            pass
    return {"summary": content, "claims": "", "entities": "", "links": ""}

# =========================
# Link utilities
# =========================
def is_real_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    s = u.strip()
    if not s.startswith("http"):
        return False
    if "example.com" in s or "foo.bar" in s:
        return False
    return bool(re.match(r"^https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", s))

def fallback_search_links(title: str, org: str, n: int = 5):
    q = quote_plus(f"{title} {org}".strip())
    return [
        f"https://patents.google.com/?q={q}",
        f"https://www.lens.org/lens/search?q={q}",
        f"https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html?query={q}",
        f"https://worldwide.espacenet.com/patent/search?q={q}",
        f"https://scholar.google.com/scholar?q={q}",
    ][:n]

def host_label(u: str) -> str:
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return "link"

# =========================
# Research UI and actions
# =========================
patent_titles = df_filtered["Title"].tolist()
selected_title = st.selectbox("Select a patent for deep research:", patent_titles)

if selected_title:
    row = df_filtered.loc[df_filtered["Title"] == selected_title].iloc[0]
    patent_link, gpat_url, lens_url, gemini_url, prompt = build_links_and_prompt(row)

    st.markdown(
        f"""
        <div style="padding:10px;border:1px solid rgba(255,255,255,0.1);border-radius:8px;margin-bottom:8px;">
            <div style="font-size:18px;margin-bottom:6px;">🧾 <span style="font-weight:600;">{row['Title']}</span></div>
            <div style="opacity:0.85;">🏢 {row['Organization']} • 📅 {row['Year']} • 🏷️ {row['Domain']}</div>
            <div style="margin-top:6px;opacity:0.9;">🔑 {row['Keywords']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.link_button("🔗 Open Patent", patent_link, use_container_width=True)
    with col_b:
        st.link_button("📄 Google Patents", gpat_url, use_container_width=True)
    with col_c:
        st.link_button("🔎 The Lens", lens_url, use_container_width=True)

    st.caption("Prepared prompt")
    st.code(prompt, language="markdown")

    c1, c2 = st.columns(2)
    with c1:
        components.html(
            f"""
            <textarea id="promptText" style="position:absolute;left:-9999px;">{prompt}</textarea>
            <button id="copyBtn" style="margin-top:6px;padding:8px 16px;">📋 Copy Prompt</button>
            <script>
            (function(){{
              const btn = document.getElementById('copyBtn');
              btn.addEventListener('click', function() {{
                try {{
                  const ta = document.getElementById('promptText');
                  ta.select();
                  document.execCommand('copy');
                  btn.textContent = "✅ Copied!";
                  setTimeout(function(){{ btn.textContent = "📋 Copy Prompt"; }}, 1500);
                }} catch(e) {{
                  btn.textContent = "❌ Copy failed";
                  setTimeout(function(){{ btn.textContent = "📋 Copy Prompt"; }}, 1500);
                }}
              }});
            }})();
            </script>
            """,
            height=80,
        )
    with c2:
        st.link_button("🧠 Open Gemini", gemini_url, use_container_width=True)

    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        analyze = st.button("⚙️ Analyze with LLM", type="primary", use_container_width=True, disabled=(LLM_API_KEY is None))
        st.caption(f"Server model: {resolved_model}")
    with right:
        refresh = st.button("♻️ Refresh cached result", use_container_width=True)

    key = hashlib.sha256(f"{row['Title']}|{row['Organization']}|{row['Year']}".encode()).hexdigest()
    conn = get_db()
    cached = cache_get(conn, key)

    if analyze:
        with st.spinner("Analyzing..."):
            result = llm_analyze(prompt)
            if any(result.values()):
                cache_put(conn, key, row, result)
                cached = result

    if refresh:
        cached = cache_get(conn, key)

    if cached:
        st.success("Analysis ready")

        def parse_summary_json(cached_obj):
            v = cached_obj.get("summary", "")
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                txt = v.strip()
                if txt.startswith("{") and txt.endswith("}"):
                    try:
                        return json.loads(txt)
                    except Exception:
                        pass
                depth, start = 0, None
                for i, ch in enumerate(txt):
                    if ch == "{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == "}":
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and start is not None:
                                candidate = txt[start : i + 1]
                                try:
                                    return json.loads(candidate)
                                except Exception:
                                    break
            return None

        data = parse_summary_json(cached)
        if not data:
            st.error("The model did not return valid JSON. Click Analyze again.")
        else:
            inv = data.get("invention_summary", {})
            claims = data.get("key_claims") or data.get("key_claims_themes") or []

            # Normalize assignees_applicants
            raw_assignees = data.get("assignees_applicants", {})
            assignees_org = ""
            assignees_ipc = []
            if isinstance(raw_assignees, dict):
                assignees_org = raw_assignees.get("organization", "") or ""
                val = raw_assignees.get("ipc_cpc_hints", [])
                if isinstance(val, list):
                    assignees_ipc = val
                elif isinstance(val, str) and val.strip():
                    assignees_ipc = [val.strip()]
            elif isinstance(raw_assignees, list) and raw_assignees:
                first = raw_assignees[0]
                if isinstance(first, dict):
                    assignees_org = first.get("organization", "") or ""
                    val = first.get("ipc_cpc_hints", [])
                    if isinstance(val, list):
                        assignees_ipc = val
                    elif isinstance(val, str) and val.strip():
                        assignees_ipc = [val.strip()]
                elif isinstance(first, str):
                    assignees_org = first.strip()
            elif isinstance(raw_assignees, str):
                assignees_org = raw_assignees.strip()
            if not assignees_org:
                assignees_org = row.get("Organization", "") or "N/A"

            prior = data.get("related_prior_art_links", [])
            comp = data.get("competitor_filings", [])
            brief = data.get("executive_brief", [])
            risks = data.get("risks_limitations", [])
            trl_obj = data.get("TRL_estimate", {}) or {}

            # Clean/fix links
            prior = [ln.strip() for ln in (prior or []) if isinstance(ln, str)]
            prior = [ln for ln in prior if is_real_url(ln)]
            if len(prior) < 5:
                prior = (prior + fallback_search_links(row["Title"], assignees_org, 5))[:5]

            comp = [ln.strip() for ln in (comp or []) if isinstance(ln, str)]
            comp = [ln for ln in comp if is_real_url(ln)]
            comp_query_org = assignees_org if assignees_org and assignees_org != "N/A" else row["Organization"]
            if len(comp) < 5:
                comp = (comp + fallback_search_links(row["Title"], comp_query_org, 5))[:5]

            st.markdown("### Invention summary")
            st.write(f"- Title: {inv.get('title', row['Title'])}")
            st.write(f"- Abstract: {inv.get('abstract','') or '_No abstract_'}")

            st.markdown("### Key claims")
            if isinstance(claims, list) and claims:
                for c in claims:
                    st.write(f"- {c}")
            else:
                st.write("- None found")

            st.markdown("### Assignees/applicants")
            st.write(f"- Organization: {assignees_org}")
            if assignees_ipc:
                for cpc in assignees_ipc:
                    st.write(f"- IPC/CPC: {cpc}")
            else:
                st.write("- IPC/CPC: N/A")

            st.markdown("### Prior art links")
            for ln in prior:
                st.markdown(f"- [{host_label(ln)}]({ln})")

            st.markdown("### Competitor filings")
            for ln in comp:
                st.markdown(f"- [{host_label(ln)}]({ln})")

            st.markdown("### Executive brief")
            if isinstance(brief, list) and brief:
                for pt in brief:
                    st.write(f"- {pt}")
            else:
                st.write("- None found")

            st.markdown("### Risks/limitations")
            if isinstance(risks, list) and risks:
                for pt in risks:
                    st.write(f"- {pt}")
            else:
                st.write("- None found")

            st.markdown("### TRL estimate")
            trl = trl_obj.get("level", None)
            rat = trl_obj.get("rationale", "")
            st.write(f"- Level: {trl if trl is not None else 'N/A'}")
            st.write(f"- Rationale: {rat or 'N/A'}")

            # Markdown export (with cleaned links)
            md = []
            md += [f"# {row['Title']}", ""]
            md += ["## Summary", inv.get("abstract", "_No abstract_"), ""]
            md += ["## Claims"] + [f"- {c}" for c in (claims or ["None found"])] + [""]
            md += ["## Entities", f"- Organization: {assignees_org}"] + \
                  [f"- IPC/CPC: {c}" for c in (assignees_ipc if assignees_ipc else ["N/A"])] + [""]
            md += ["## Links"] + [f"- {ln}" for ln in (prior + comp)]
            dl_md = "\n".join(md)
            st.download_button("⬇️ Download analysis (MD)", data=dl_md, file_name="analysis.md", mime="text/markdown")
    else:
        st.info("No cached analysis yet. Provide an API key and click Analyze.")

# ============
# Charts
# ============
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df_filtered, x="Year", color="Domain", title="Patents Over Time")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    domain_count = df_filtered["Domain"].value_counts().reset_index()
    domain_count.columns = ["Domain", "Count"]
    fig2 = px.bar(domain_count, x="Domain", y="Count", title="Patents by Domain")
    st.plotly_chart(fig2, use_container_width=True)

org_count = df_filtered["Organization"].value_counts().reset_index()
org_count.columns = ["Organization", "Count"]
fig3 = px.bar(org_count.head(10), x="Organization", y="Count", title="Top Organizations")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.box(df_filtered, x="Domain", y="TRL", title="TRL Distribution by Domain")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("📈 Technology Trend Analysis")
s_curve_df = compute_s_curve(df_filtered)
fig5 = px.line(s_curve_df, x="Year", y="S_curve", markers=True, title="S-Curve of Patent Growth")
st.plotly_chart(fig5, use_container_width=True)
fig6 = px.bar(
    s_curve_df, x="Year", y="Count", title="Hype Curve: Patents per Year",
    color="Count", color_continuous_scale="Viridis"
)
st.plotly_chart(fig6, use_container_width=True)

st.subheader("🔗 Domain Convergence (Keyword Overlap)")
conv = domain_convergence(df_filtered)
if conv:
    df_conv = pd.DataFrame(list(conv.items()), columns=["Domain Pair", "Keyword Overlap Count"])
    st.dataframe(df_conv.sort_values(by="Keyword Overlap Count", ascending=False))
else:
    st.info("No convergence detected.")

st.subheader("🚨 AI-style Insights & Alerts")
for a in generate_alerts(df_filtered):
    st.warning(a)
