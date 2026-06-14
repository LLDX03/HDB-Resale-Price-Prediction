"""
STEP 1 — LOAD & EXPLORE HDB DATA
=================================
Run this first. It loads your CSV, shows you the structure,
checks for missing values, and prints key stats.

HOW TO RUN:
    python src/01_load_data.py
"""

import pandas as pd
import os

# ── Load the CSV ────────────────────────────────────────────────────────────
# Make sure your file is at: data/hdb.csv
# If you get a FileNotFoundError, check the path below matches your file location.

DATA_PATH = "data/hdb.csv"

if not os.path.exists(DATA_PATH):
    print(f"❌  ERROR: File not found at '{DATA_PATH}'")
    print("    Fix: Move your CSV into the 'data/' folder and rename it hdb.csv")
    exit()

df = pd.read_csv(DATA_PATH)
print(f"✅  Loaded {len(df):,} rows and {len(df.columns)} columns\n")

# ── What columns do we have? ────────────────────────────────────────────────
print("=== COLUMN NAMES ===")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}")

# ── Data types ──────────────────────────────────────────────────────────────
print("\n=== DATA TYPES ===")
print(df.dtypes.to_string())

# ── Missing values ──────────────────────────────────────────────────────────
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅  No missing values! Clean dataset.")
else:
    print(missing[missing > 0])

# ── First 5 rows ────────────────────────────────────────────────────────────
print("\n=== FIRST 5 ROWS ===")
print(df.head())

# ── Price stats ─────────────────────────────────────────────────────────────
print("\n=== RESALE PRICE STATISTICS (S$) ===")
stats = df['resale_price'].describe()
print(f"  Count:    {stats['count']:,.0f} transactions")
print(f"  Min:      S${stats['min']:,.0f}")
print(f"  Max:      S${stats['max']:,.0f}")
print(f"  Average:  S${stats['mean']:,.0f}")
print(f"  Median:   S${stats['50%']:,.0f}")
print(f"  Std Dev:  S${stats['std']:,.0f}")

# ── Unique values ────────────────────────────────────────────────────────────
print("\n=== UNIQUE VALUES ===")
print(f"  Towns:      {df['town'].nunique()} unique towns")
print(f"  Flat types: {df['flat_type'].nunique()} → {sorted(df['flat_type'].unique())}")
print(f"  Years:      {df['month'].str[:4].unique().tolist()}")