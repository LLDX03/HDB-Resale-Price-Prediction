"""
HDB Resale Price Predictor — Streamlit Web App
HOW TO RUN:  python -m streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="wide")

SQM_TO_SQFT = 10.7639
ANNUAL_GROWTH_RATE = 0.045

# ── Hard limits per flat type — drives both the input widget AND validation ───
# Based on actual HDB resale transaction data (data.gov.sg)
FLAT_LIMITS = {
    "1 ROOM":    {"min": 25,  "max": 45,  "default": 33,  "typical": "25–45 sqm  (269–484 sqft)"},
    "2 ROOM":    {"min": 34,  "max": 50,  "default": 40,  "typical": "36–46 sqm  (388–495 sqft)"},
    "3 ROOM":    {"min": 55,  "max": 85,  "default": 68,  "typical": "60–75 sqm  (646–807 sqft)"},
    "4 ROOM":    {"min": 80,  "max": 115, "default": 95,  "typical": "85–105 sqm  (915–1,130 sqft)"},
    "5 ROOM":    {"min": 100, "max": 140, "default": 118, "typical": "105–130 sqm  (1,130–1,400 sqft)"},
    "EXECUTIVE": {"min": 125, "max": 170, "default": 145, "typical": "130–160 sqm  (1,400–1,722 sqft)"},
    "MULTI-GENERATION": {"min": 155, "max": 200, "default": 170, "typical": "155–200 sqm  (1,668–2,153 sqft)"},
}

FLAT_MODEL_INFO = {
    "Improved": {
        "emoji": "🏗️", "era": "1970s – 1980s",
        "desc": "One of HDB's earliest designs. Smaller rooms, lower ceilings, open kitchen with no dedicated dining area. Toilet and bathroom are often separate. Usually the most affordable model on the resale market.",
        "size": "~45–75 sqm depending on flat type",
        "note": "Many units have gone through HDB's Home Improvement Programme (HIP), upgrading toilets and common areas.",
    },
    "New Generation": {
        "emoji": "🔄", "era": "1980s",
        "desc": "An upgrade over the Improved model. Introduced a dedicated living and dining area, slightly better ventilation, and a more practical layout.",
        "size": "~69–83 sqm (3-room); ~100–105 sqm (4-room)",
        "note": "Master bedroom in 3-room NG comes with an attached bathroom — unusual for that era.",
    },
    "Model A": {
        "emoji": "📐", "era": "1990s",
        "desc": "The most common HDB design in the resale market today. Enclosed kitchen, proper dining area, better ventilation, and larger bedrooms.",
        "size": "~75 sqm (3-room); ~105 sqm (4-room); ~135 sqm (5-room)",
        "note": "Widely available across all estates — the default choice for most buyers.",
    },
    "Standard": {
        "emoji": "🏠", "era": "1980s – 1990s",
        "desc": "A cost-optimised design, slightly smaller than Model A. Simpler layout with no attached bathroom in the master bedroom for most 3-room units.",
        "size": "~54–65 sqm (3-room); ~84 sqm (4-room)",
        "note": "Also recorded as 'Simplified' in some HDB documents.",
    },
    "Premium Apartment": {
        "emoji": "✨", "era": "1990s – 2000s",
        "desc": "HDB's higher-end design built to near-condominium standards. Taller ceilings (up to 2.8m), larger windows, better-quality fittings, more generous room proportions.",
        "size": "Similar sqm to Model A but with premium finishes",
        "note": "DBSS (Design, Build and Sell Scheme) flats built by private developers also fall under this category.",
    },
    "Maisonette": {
        "emoji": "🏘️", "era": "1984 – 2000 (discontinued)",
        "desc": "A rare two-storey unit with a staircase inside the flat. Lower level has living, dining, kitchen, and shared bath. Upper level has 3 bedrooms with master en-suite. 1.5–2× larger than a typical 5-room flat.",
        "size": "137–243 sqm (1,475–2,616 sqft)",
        "note": "No new Maisonettes built since 1995. Highly sought after for the 'landed home feel' within HDB.",
    },
}

def show_flat_model_info(selected_model):
    if selected_model in FLAT_MODEL_INFO:
        info = FLAT_MODEL_INFO[selected_model]
        st.info(
            f"{info['emoji']} **{selected_model}** &nbsp;|&nbsp; "
            f"Built: {info['era']} &nbsp;|&nbsp; Size: {info['size']}\n\n"
            f"{info['desc']}\n\n💡 *{info['note']}*"
        )
    else:
        st.info(f"🔧 **{selected_model}** — Check HDB's website for details on this model.")

def get_area_limits(flat_type, unit):
    """Return (min, max, default) in the chosen unit."""
    if flat_type in FLAT_LIMITS:
        lim = FLAT_LIMITS[flat_type]
        if unit == "sqft":
            return (
                round(lim["min"] * SQM_TO_SQFT),
                round(lim["max"] * SQM_TO_SQFT),
                round(lim["default"] * SQM_TO_SQFT),
                lim["typical"],
            )
        return lim["min"], lim["max"], lim["default"], lim["typical"]
    # Fallback — should never hit if all flat types are covered
    if unit == "sqft":
        return 200, 2200, 1000, "—"
    return 20, 200, 90, "—"

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/rf_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model_data = load_model()

st.title("🏠 HDB Resale Price Predictor")
st.markdown("**Singapore HDB Flat Price Prediction using Machine Learning**")
st.markdown("---")

if model_data is None:
    st.error("⚠️  Model not found. Run `python src/04_train_model.py` first.")
    st.stop()

rf, le_town, le_flat, le_model = (
    model_data["rf"], model_data["le_town"],
    model_data["le_flat"], model_data["le_model"],
)

tab1, tab2 = st.tabs(["🔮 Predict Current Price", "📈 Future Price Projection"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Flat Details")
    col1, col2 = st.columns(2)

    with col1:
        town      = st.selectbox("📍 Town", sorted(le_town.classes_))
        flat_type = st.selectbox("🏢 Flat Type", sorted(le_flat.classes_))
        year      = st.slider("📅 Transaction Year", 2017, 2024, 2024)
        storey    = st.slider("🏗️ Floor Level (storey)", 1, 40, 8)

    with col2:
        remaining_lease = st.slider("📋 Remaining Lease (years)", 40, 99, 70)
        area_unit = st.radio("📐 Floor Area Unit", ["sqm (metric)", "sqft (imperial)"],
                             horizontal=True)
        unit_key = "sqft" if "sqft" in area_unit else "sqm"

        # Limits change dynamically with flat type selection
        a_min, a_max, a_default, a_typical = get_area_limits(flat_type, unit_key)
        unit_label = "sqft" if unit_key == "sqft" else "sqm"

        floor_area_input = st.number_input(
            f"Floor Area ({unit_label})",
            min_value=float(a_min),
            max_value=float(a_max),
            value=float(a_default),
            step=5.0 if unit_key == "sqft" else 0.5,
            help=f"Valid range for {flat_type}: {a_typical}",
        )

        # Convert to sqm for the model; show the other unit as info
        if unit_key == "sqft":
            floor_area_sqm  = floor_area_input / SQM_TO_SQFT
            floor_area_sqft = floor_area_input
            st.caption(f"≈ {floor_area_sqm:.1f} sqm  |  Valid for {flat_type}: {a_typical}")
        else:
            floor_area_sqm  = floor_area_input
            floor_area_sqft = floor_area_input * SQM_TO_SQFT
            st.caption(f"≈ {floor_area_sqft:.0f} sqft  |  Valid for {flat_type}: {a_typical}")

        flat_model = st.selectbox("🔧 Flat Model", sorted(le_model.classes_))

    show_flat_model_info(flat_model)
    st.caption("ℹ️ Flat model descriptions sourced from ERA Singapore & PropNex property guides.")
    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True, key="btn1"):
        # Secondary hard check (in case min/max widget limits are bypassed)
        a_min_sqm = FLAT_LIMITS.get(flat_type, {}).get("min", 0)
        a_max_sqm = FLAT_LIMITS.get(flat_type, {}).get("max", 9999)

        if floor_area_sqm < a_min_sqm or floor_area_sqm > a_max_sqm:
            st.error(
                f"❌ **{floor_area_sqm:.1f} sqm is not a valid size for a {flat_type} flat.**\n\n"
                f"Valid range: **{a_min_sqm}–{a_max_sqm} sqm** ({a_typical})\n\n"
                "Please adjust the floor area and try again."
            )
        else:
            warnings = []
            if remaining_lease < 55:
                warnings.append(f"⚠️ Short lease ({remaining_lease} yrs) — CPF and HDB loan eligibility may be restricted.")
            if storey > 30 and flat_type in ["1 ROOM", "2 ROOM"]:
                warnings.append(f"⚠️ {flat_type} flats above storey 30 are uncommon. Prediction may be less accurate.")
            for w in warnings:
                st.warning(w)

            try:
                input_data = pd.DataFrame([{
                    "floor_area_sqm":        floor_area_sqm,
                    "storey_mid":            storey,
                    "remaining_lease_years": remaining_lease,
                    "year":                  year,
                    "town_enc":              le_town.transform([town])[0],
                    "flat_type_enc":         le_flat.transform([flat_type])[0],
                    "flat_model_enc":        le_model.transform([flat_model])[0],
                }])
                prediction = rf.predict(input_data)[0]
                low, high = prediction * 0.90, prediction * 1.10

                st.success(f"### 💰 Predicted Resale Price: **S${prediction:,.0f}**")
                st.info(f"📊 Estimated range: **S${low:,.0f} – S${high:,.0f}** (±10%)")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Per sqm",   f"S${prediction/floor_area_sqm:,.0f}")
                m2.metric("Per sqft",  f"S${prediction/floor_area_sqft:,.0f}")
                m3.metric("Flat Type", flat_type)
                m4.metric("Town",      town)

                st.caption("⚠️ Estimate only. Actual prices vary by market conditions, "
                           "flat condition, renovation, and negotiation.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Future Price Projection")
    st.markdown("Projects price forward using historical HDB growth (~4.5%/yr), adjusted for lease decay.")

    col3, col4 = st.columns(2)
    with col3:
        f_town       = st.selectbox("📍 Town", sorted(le_town.classes_), key="ft")
        f_flat_type  = st.selectbox("🏢 Flat Type", sorted(le_flat.classes_), key="ff")
        f_flat_model = st.selectbox("🔧 Flat Model", sorted(le_model.classes_), key="fm")
        f_storey     = st.slider("🏗️ Floor Level", 1, 40, 8, key="fs")

    with col4:
        f_lease    = st.slider("📋 Remaining Lease (years)", 40, 99, 70, key="fl")
        f_unit_sel = st.radio("📐 Floor Area Unit", ["sqm (metric)", "sqft (imperial)"],
                              horizontal=True, key="fu")
        f_unit_key = "sqft" if "sqft" in f_unit_sel else "sqm"

        fa_min, fa_max, fa_default, fa_typical = get_area_limits(f_flat_type, f_unit_key)
        f_unit_label = "sqft" if f_unit_key == "sqft" else "sqm"

        f_area_input = st.number_input(
            f"Floor Area ({f_unit_label})",
            min_value=float(fa_min), max_value=float(fa_max),
            value=float(fa_default),
            step=5.0 if f_unit_key == "sqft" else 0.5,
            key="fa",
            help=f"Valid range for {f_flat_type}: {fa_typical}",
        )
        if f_unit_key == "sqft":
            f_sqm  = f_area_input / SQM_TO_SQFT
            f_sqft = f_area_input
            st.caption(f"≈ {f_sqm:.1f} sqm  |  Valid for {f_flat_type}: {fa_typical}")
        else:
            f_sqm  = f_area_input
            f_sqft = f_area_input * SQM_TO_SQFT
            st.caption(f"≈ {f_sqft:.0f} sqft  |  Valid for {f_flat_type}: {fa_typical}")

        proj_years = st.slider("📅 Project how many years ahead?", 1, 15, 5, key="fy")

    show_flat_model_info(f_flat_model)
    st.caption("ℹ️ Flat model descriptions sourced from ERA Singapore & PropNex property guides.")
    st.markdown("---")

    if st.button("📈 Project Future Price", type="primary", use_container_width=True, key="btn2"):
        fa_min_sqm = FLAT_LIMITS.get(f_flat_type, {}).get("min", 0)
        fa_max_sqm = FLAT_LIMITS.get(f_flat_type, {}).get("max", 9999)

        if f_sqm < fa_min_sqm or f_sqm > fa_max_sqm:
            st.error(
                f"❌ **{f_sqm:.1f} sqm is not a valid size for a {f_flat_type} flat.**\n\n"
                f"Valid range: **{fa_min_sqm}–{fa_max_sqm} sqm** ({fa_typical})\n\n"
                "Please adjust the floor area and try again."
            )
        else:
            try:
                base_input = pd.DataFrame([{
                    "floor_area_sqm":        f_sqm,
                    "storey_mid":            f_storey,
                    "remaining_lease_years": f_lease,
                    "year":                  2024,
                    "town_enc":              le_town.transform([f_town])[0],
                    "flat_type_enc":         le_flat.transform([f_flat_type])[0],
                    "flat_model_enc":        le_model.transform([f_flat_model])[0],
                }])
                base_price = rf.predict(base_input)[0]

                years, prices, p_low, p_hi = [], [], [], []
                for i in range(proj_years + 1):
                    lease_factor = (max(f_lease - i, 1) / f_lease) ** 0.3
                    p = base_price * ((1 + ANNUAL_GROWTH_RATE) ** i) * lease_factor
                    years.append(2024 + i); prices.append(p)
                    p_low.append(p * 0.90); p_hi.append(p * 1.10)

                target = prices[-1]
                gain   = target - base_price

                st.success(f"### 💰 Projected Price in {2024+proj_years}: **S${target:,.0f}**")
                g1, g2, g3 = st.columns(3)
                g1.metric("Current (2024)",      f"S${base_price:,.0f}")
                g2.metric(f"In {2024+proj_years}", f"S${target:,.0f}", delta=f"+S${gain:,.0f}")
                g3.metric("Total Growth",         f"+{gain/base_price*100:.1f}%")

                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor("#F1FAEE"); ax.set_facecolor("#F1FAEE")
                ax.fill_between(years, [p/1000 for p in p_low], [p/1000 for p in p_hi],
                                alpha=0.2, color="#457B9D", label="±10% range")
                ax.plot(years, [p/1000 for p in prices], "o-", color="#457B9D",
                        linewidth=2.5, markersize=7, label="Projected Price")
                ax.axvline(2024, color="gray", linestyle="--", alpha=0.5, label="Today (2024)")
                for yr, pr in zip(years, prices):
                    ax.annotate(f"S${pr/1000:.0f}K", (yr, pr/1000),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=8.5, color="#1D3557")
                ax.set_xlabel("Year", fontsize=12); ax.set_ylabel("Price (S$000)", fontsize=12)
                ax.set_title(f"{f_flat_type} in {f_town} — {proj_years}-year Price Projection",
                             fontsize=13, fontweight="bold", color="#1D3557")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.grid(alpha=0.3, linestyle="--"); ax.legend(fontsize=10)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.caption(f"⚠️ Assumes ~{ANNUAL_GROWTH_RATE*100:.1f}% annual growth with lease decay adjustment. "
                           "Not financial advice.")
            except Exception as e:
                st.error(f"Projection error: {e}")

st.markdown("---")
st.markdown("**Model:** Random Forest (R² = 0.843)  |  **Data:** HDB Resale 2017–2024  |  "
            "**Flat model info:** ERA Singapore, PropNex")