import pandas as pd
import glob
import random
import re

# --------------------------
# 1️⃣ Get all CSV files in 'data' folder
# --------------------------
csv_files = glob.glob('data/*.csv')

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    
    # --------------------------
    # 2️⃣ Normalize column names
    # --------------------------
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '', col).lower() for col in df.columns]
    
    # --------------------------
    # 3️⃣ Detect columns robustly
    # --------------------------
    # Title
    title_col = next((col for col in df.columns if 'title' in col), None)
    
    # Year
    year_col = next((col for col in df.columns if 'year' in col), None)
    
    # Organization / Assignee
    org_col = next((col for col in df.columns if 'assignee' in col or 'applicant' in col or 'organization' in col), None)
    
    # Keywords / Abstract
    kw_col = next((col for col in df.columns if 'abstract' in col or 'keyword' in col or 'summary' in col), None)
    
    # --------------------------
    # 4️⃣ Build clean dataframe
    # --------------------------
    df_clean = pd.DataFrame()
    
    df_clean['Title'] = df[title_col] if title_col else "Unknown Title"
    df_clean['Domain'] = df['domain'] if 'domain' in df.columns else file.split('\\')[-1].split('.')[0]
    df_clean['Year'] = df[year_col] if year_col else 2023
    
    # Organization (join multiple names if present)
    if org_col:
        df_clean['Organization'] = df[org_col].apply(lambda x: str(x).replace(';', ',').strip())
    else:
        df_clean['Organization'] = "Unknown Org"
    
    df_clean['Keywords'] = df[kw_col] if kw_col else "N/A"
    
    # TRL column
    if 'trl' in df.columns:
        df_clean['TRL'] = df['trl']
    else:
        df_clean['TRL'] = [random.randint(1,9) for _ in range(len(df_clean))]
    
    # Optional: limit rows per CSV for demo
    df_clean = df_clean.head(200)
    
    all_dfs.append(df_clean)

# --------------------------
# 5️⃣ Merge all CSVs
# --------------------------
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.drop_duplicates(inplace=True)
final_df.reset_index(drop=True, inplace=True)

# --------------------------
# 6️⃣ Save cleaned CSV
# --------------------------
final_df.to_csv('data/multi_domain_demo_data_clean.csv', index=False)

print("✅ Clean CSV ready! Total rows:", len(final_df))
