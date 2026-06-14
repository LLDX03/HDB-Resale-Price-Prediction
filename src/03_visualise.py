"""
STEP 3 — VISUALISATIONS
========================
Creates 5 charts and saves them to outputs

HOW TO RUN:
    python src/03_visualise.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/hdb.csv")
df['year'] = df['month'].str[:4].astype(int)
df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+)').astype(int)
df['storey_mid'] = df['storey_range'].str.extract(r'(\d+) TO').astype(int) + 1

# ── Style settings ───────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})
BLUE = '#457B9D'; ACCENT = '#E63946'; DARK = '#1D3557'; SAND = '#F1FAEE'

# ── CHART 1: Price Distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['resale_price'] / 1000, bins=60, color=BLUE, edgecolor='white', linewidth=0.4)
ax.axvline(df['resale_price'].median() / 1000, color=ACCENT, linewidth=2, linestyle='--',
           label=f"Median: S${df['resale_price'].median()/1000:.0f}K")
ax.axvline(df['resale_price'].mean() / 1000, color=DARK, linewidth=2, linestyle=':',
           label=f"Mean: S${df['resale_price'].mean()/1000:.0f}K")
ax.set_xlabel('Resale Price (S$000)', fontsize=12)
ax.set_ylabel('Number of Transactions', fontsize=12)
ax.set_title('HDB Resale Price Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/01_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  Chart 1 saved: Price Distribution")

# ── CHART 2: Average Price by Town ───────────────────────────────────────────
town_avg = df.groupby('town')['resale_price'].mean().sort_values(ascending=True) / 1000
median_price = town_avg.median()
bar_colors = [ACCENT if p > median_price else BLUE for p in town_avg.values]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(town_avg.index, town_avg.values, color=bar_colors, edgecolor='white')
for bar, val in zip(bars, town_avg.values):
    ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
            f'S${val:.0f}K', va='center', fontsize=8.5, fontweight='bold')
ax.axvline(median_price, color='gray', linestyle='--', alpha=0.6,
           label=f'Median: S${median_price:.0f}K')
ax.set_xlabel('Average Resale Price (S$000)', fontsize=12)
ax.set_title('Average HDB Resale Price by Town  (Red = Above Median)', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/02_price_by_town.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  Chart 2 saved: Price by Town")

# ── CHART 3: Floor Area vs Price (Scatter) ───────────────────────────────────
flat_colors = {'2 ROOM': '#E63946', '3 ROOM': '#457B9D', '4 ROOM': '#1D3557',
               '5 ROOM': '#2A9D8F', 'EXECUTIVE': '#E9C46A'}
fig, ax = plt.subplots(figsize=(10, 6))
for ft, color in flat_colors.items():
    mask = df['flat_type'] == ft
    ax.scatter(df.loc[mask, 'floor_area_sqm'], df.loc[mask, 'resale_price'] / 1000,
               c=color, label=ft, alpha=0.35, s=15, edgecolors='none')

# Add trend line
z = np.polyfit(df['floor_area_sqm'], df['resale_price'] / 1000, 1)
x_line = np.linspace(df['floor_area_sqm'].min(), df['floor_area_sqm'].max(), 200)
ax.plot(x_line, np.poly1d(z)(x_line), color=DARK, linewidth=2.5, label='Trend Line')

ax.set_xlabel('Floor Area (sqm)', fontsize=12)
ax.set_ylabel('Resale Price (S$000)', fontsize=12)
ax.set_title('Floor Area vs Resale Price', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/03_area_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  Chart 3 saved: Area vs Price Scatter")

# ── CHART 4: Price Trend Over Time ───────────────────────────────────────────
yearly = df.groupby('year')['resale_price'].agg(['mean', 'median']).reset_index() / 1000
yearly['year'] = df['year'].unique()[:len(yearly)]  # fix year column

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(yearly['year'], yearly['mean'], alpha=0.15, color=BLUE)
ax.plot(yearly['year'], yearly['mean'], 'o-', color=BLUE, linewidth=2.5,
        markersize=7, label='Mean Price')
ax.plot(yearly['year'], yearly['median'], 's--', color=ACCENT, linewidth=2,
        markersize=7, label='Median Price')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Price (S$000)', fontsize=12)
ax.set_title('HDB Resale Price Trend Over Time', fontsize=13, fontweight='bold')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend()
plt.tight_layout()
plt.savefig('outputs/04_price_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  Chart 4 saved: Price Trend")

# ── CHART 5: Correlation Heatmap ────────────────────────────────────────────
corr_cols = ['floor_area_sqm', 'storey_mid', 'remaining_lease_years', 'year', 'resale_price']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅  Chart 5 saved: Correlation Heatmap")

print("\n🎉  All charts saved to outputs/")