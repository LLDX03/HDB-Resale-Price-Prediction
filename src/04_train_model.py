"""
STEP 4 — MACHINE LEARNING MODEL
=================================
Trains two models and compares them:
  1. Linear Regression  (simple baseline)
  2. Random Forest      (better accuracy)

Saves the best model to: models/rf_model.pkl

HOW TO RUN:
    python src/04_train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

# ── STEP 1: Load & feature engineer ─────────────────────────────────────────
df = pd.read_csv("data/hdb.csv")
df['year'] = df['month'].str[:4].astype(int)
df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+)').astype(int)
df['storey_mid'] = df['storey_range'].str.extract(r'(\d+) TO').astype(int) + 1
print(f"Loaded {len(df):,} rows")

# ── STEP 2: Encode categorical columns ──────────────────────────────────────
# Machine learning can't use text directly — we convert text → numbers
# LabelEncoder: ['ANG MO KIO', 'BEDOK', ...] → [0, 1, 2, ...]

le_town  = LabelEncoder()
le_flat  = LabelEncoder()
le_model = LabelEncoder()

df['town_enc']       = le_town.fit_transform(df['town'])
df['flat_type_enc']  = le_flat.fit_transform(df['flat_type'])
df['flat_model_enc'] = le_model.fit_transform(df['flat_model'])

print("\nEncoding examples:")
for town, code in zip(le_town.classes_[:4], range(4)):
    print(f"  '{town}' → {code}")

# ── STEP 3: Choose features and target ──────────────────────────────────────
# Features = inputs the model uses to make predictions
# Target = what we want to predict (resale price)

FEATURES = [
    'floor_area_sqm',       # Size of flat
    'storey_mid',           # Floor level
    'remaining_lease_years',# Lease left
    'year',                 # Transaction year
    'town_enc',             # Location (encoded)
    'flat_type_enc',        # Flat type (encoded)
    'flat_model_enc',       # Flat model (encoded)
]
TARGET = 'resale_price'

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeatures used: {FEATURES}")
print(f"X shape: {X.shape}  (rows × features)")
print(f"y shape: {y.shape}  (target values)")

# ── STEP 4: Train/Test Split ─────────────────────────────────────────────────
# 80% of data → train the model
# 20% of data → test if predictions are accurate (model has NEVER seen these)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing  samples: {len(X_test):,}")

# ── STEP 5: Model 1 — Linear Regression ─────────────────────────────────────
print("\n─── Training Linear Regression ───")
lr = LinearRegression()
lr.fit(X_train, y_train)        # Learn from training data
y_pred_lr = lr.predict(X_test)  # Predict on test data

lr_mae  = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2   = r2_score(y_test, y_pred_lr)

print(f"  MAE:  S${lr_mae:,.0f}")
print(f"        → On average, predictions are off by S${lr_mae:,.0f}")
print(f"  RMSE: S${lr_rmse:,.0f}")
print(f"  R²:   {lr_r2:.4f}  ({lr_r2*100:.1f}% of price variation explained)")

# ── STEP 6: Model 2 — Random Forest ─────────────────────────────────────────
print("\n─── Training Random Forest (this takes ~30 seconds) ───")
rf = RandomForestRegressor(
    n_estimators=200,   # 200 decision trees
    max_depth=12,       # Each tree goes 12 levels deep
    min_samples_leaf=4, # Each leaf needs at least 4 samples
    n_jobs=-1,          # Use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2   = r2_score(y_test, y_pred_rf)

print(f"  MAE:  S${rf_mae:,.0f}")
print(f"        → On average, predictions are off by S${rf_mae:,.0f}")
print(f"  RMSE: S${rf_rmse:,.0f}")
print(f"  R²:   {rf_r2:.4f}  ({rf_r2*100:.1f}% of price variation explained)")

# ── STEP 7: Compare models ──────────────────────────────────────────────────
print("\n=== MODEL COMPARISON ===")
print(f"{'Metric':<12} {'Linear Regression':>20} {'Random Forest':>20}")
print("-" * 55)
print(f"{'MAE':<12} {'S$'+f'{lr_mae:,.0f}':>20} {'S$'+f'{rf_mae:,.0f}':>20}")
print(f"{'RMSE':<12} {'S$'+f'{lr_rmse:,.0f}':>20} {'S$'+f'{rf_rmse:,.0f}':>20}")
print(f"{'R²':<12} {lr_r2:>20.3f} {rf_r2:>20.3f}")
print(f"\n→ Random Forest is S${lr_mae-rf_mae:,.0f} more accurate per prediction")

# ── STEP 8: Feature Importance ──────────────────────────────────────────────
print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:30} {bar} {imp:.3f}")

# ── STEP 9: Save the model ──────────────────────────────────────────────────
model_data = {
    'rf':       rf,
    'le_town':  le_town,
    'le_flat':  le_flat,
    'le_model': le_model,
    'features': FEATURES,
}
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("\n✅  Model saved to models/rf_model.pkl")

# ── STEP 10: Quick prediction example ───────────────────────────────────────
print("\n=== EXAMPLE PREDICTION ===")
example = pd.DataFrame([{
    'floor_area_sqm':       95.0,
    'storey_mid':           8,
    'remaining_lease_years':70,
    'year':                 2024,
    'town_enc':             le_town.transform(['TAMPINES'])[0],
    'flat_type_enc':        le_flat.transform(['4 ROOM'])[0],
    'flat_model_enc':       le_model.transform(['Model A'])[0],
}])
pred = rf.predict(example)[0]
print(f"  Input: 4-room flat in Tampines, 95sqm, storey 8, 70 years lease, 2024")
print(f"  Predicted Price: S${pred:,.0f}")

print("\n🎉  Training complete!")