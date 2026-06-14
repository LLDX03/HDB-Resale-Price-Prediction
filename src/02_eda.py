"""
STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
==========================================
Analyse patterns in the HDB data:
  - Price by town
  - Price by flat type
  - Trends over time

HOW TO RUN:
    python src/02_eda.py
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/hdb.csv")

# ── Feature engineering ─────────────────────────────────────────────────────
# Extract year from "2022-03" format
df['year'] = df['month'].str[:4].astype(int)

# Extract number from "65 years" → 65
df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+)').astype(int)

# Extract floor number from "07 TO 09" → 8
df['storey_mid'] = df['storey_range'].str.extract(r'(\d+) TO').astype(int) + 1

print("=== AVERAGE PRICE BY TOWN (sorted highest → lowest) ===")
town_avg = (df.groupby('town')['resale_price']
              .mean()
              .sort_values(ascending=False)
              .reset_index())
town_avg.columns = ['Town', 'Avg Price (S$)']
town_avg['Avg Price (S$)'] = town_avg['Avg Price (S$)'].apply(lambda x: f"S${x:,.0f}")
print(town_avg.to_string(index=False))

print("\n=== AVERAGE PRICE BY FLAT TYPE ===")
flat_avg = (df.groupby('flat_type')['resale_price']
              .agg(['mean', 'median', 'count'])
              .sort_values('mean', ascending=False)
              .reset_index())
flat_avg.columns = ['Flat Type', 'Mean Price', 'Median Price', 'Transactions']
flat_avg['Mean Price'] = flat_avg['Mean Price'].apply(lambda x: f"S${x:,.0f}")
flat_avg['Median Price'] = flat_avg['Median Price'].apply(lambda x: f"S${x:,.0f}")
print(flat_avg.to_string(index=False))

print("\n=== PRICE TREND BY YEAR ===")
year_trend = (df.groupby('year')['resale_price']
                .agg(['mean', 'median', 'count'])
                .reset_index())
year_trend.columns = ['Year', 'Mean Price', 'Median Price', 'Transactions']
for col in ['Mean Price', 'Median Price']:
    year_trend[col] = year_trend[col].apply(lambda x: f"S${x:,.0f}")
print(year_trend.to_string(index=False))

print("\n=== CORRELATION WITH RESALE PRICE ===")
numeric_cols = ['floor_area_sqm', 'storey_mid', 'remaining_lease_years', 'year']
corr = df[numeric_cols + ['resale_price']].corr()['resale_price'].drop('resale_price')
corr_sorted = corr.abs().sort_values(ascending=False)
print("  (1.0 = perfect correlation, 0 = no correlation)")
for col, val in corr_sorted.items():
    direction = "↑ rises" if corr[col] > 0 else "↓ falls"
    print(f"  {col:30} {val:.3f}  (price {direction} with this)")

print("\n=== KEY INSIGHTS ===")
print(f"  • Most expensive town: {df.groupby('town')['resale_price'].mean().idxmax()}")
print(f"  • Most affordable town: {df.groupby('town')['resale_price'].mean().idxmin()}")
print(f"  • Price growth 2017→2024: ", end="")
y17 = df[df['year']==2017]['resale_price'].mean()
y24 = df[df['year']==2024]['resale_price'].mean()
print(f"S${y17:,.0f} → S${y24:,.0f} (+{(y24-y17)/y17*100:.1f}%)")
print(f"  • Floor area correlation with price: {df['floor_area_sqm'].corr(df['resale_price']):.3f}")