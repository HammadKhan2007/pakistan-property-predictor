import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan Property Price Predictor", layout="wide")

st.title("🏡 Pakistan Property Price Predictor")
st.markdown("Predict market price ranges using XGBoost trained on 1.4 Lakh+ Zameen.com listings. SHAP explainability in PKR.")

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_assets():
    if not os.path.exists('models/xgb_model.json'):
        return None
    m = xgb.XGBRegressor()
    m.load_model('models/xgb_model.json')
    # Safely load SHAP explainer
    shap_explainer = None
    try:
        shap_explainer = joblib.load('models/shap_explainer_raw.pkl')
    except Exception:
        try:
            shap_explainer = shap.TreeExplainer(m)
        except Exception:
            shap_explainer = None

    return {
        'model': m,
        'q_lower': joblib.load('models/lgb_lower.pkl'),
        'q_upper': joblib.load('models/lgb_upper.pkl'),
        'features': joblib.load('models/features.pkl'),
        'cat_cols': joblib.load('models/cat_cols.pkl'),
        'encoder': joblib.load('models/cat_encoder.pkl'),
        'loc_data': joblib.load('models/locations.pkl'),
        'shap_explainer': shap_explainer,
    }

assets = load_assets()

if assets is None:
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

loc_data = assets['loc_data']
cities = loc_data['cities']
locations_dict = loc_data['locations']
society_avg_map = loc_data['society_avg_map']
global_median = loc_data.get('global_median', 1e6)

CITY_CENTERS = {
    'lahore': (31.5204, 74.3587), 'karachi': (24.8607, 67.0011),
    'islamabad': (33.6844, 73.0479), 'rawalpindi': (33.5651, 73.0169),
    'faisalabad': (31.4504, 73.1350),
}

prop_types = ['House', 'Flat', 'Lower Portion', 'Upper Portion', 'Penthouse', 'Farm House', 'Room']

def format_pkr(amount):
    if amount >= 1e7:
        return f"PKR {amount/1e7:.2f} Crore"
    elif amount >= 1e5:
        return f"PKR {amount/1e5:.2f} Lakh"
    return f"PKR {amount:,.0f}"

# ============================================================================
# INPUT FORM
# ============================================================================
st.markdown("### 📝 Enter Property Details")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", [c.title() for c in cities])
    prop_type = st.selectbox("Property Type", prop_types)
with col2:
    city_locs = locations_dict.get(city.lower(), ['Other'])
    location = st.selectbox("Location", [str(l).title() for l in city_locs])
    c_area, c_unit = st.columns([2, 1])
    with c_area:
        area_num = st.number_input("Area", min_value=0.5, value=5.0, step=0.5)
    with c_unit:
        area_unit = st.selectbox("Unit", ["Marla", "Kanal", "Sq. Yd.", "Sq. Ft."])

col3, col4 = st.columns(2)
with col3:
    bedrooms = st.number_input("Bedrooms", 0, 20, 3)
with col4:
    baths = st.number_input("Bathrooms", 0, 20, 3)

# Convert area
area_marla = area_num
if area_unit == "Kanal":
    area_marla = area_num * 20
elif area_unit == "Sq. Yd.":
    area_marla = area_num / 25
elif area_unit == "Sq. Ft.":
    area_marla = area_num / 225

st.markdown("---")

if st.button("🔍 Predict Price", type="primary"):
    # Build features
    soc_avg = society_avg_map.get((city.lower(), location.lower()), global_median)
    bpb = baths / bedrooms if bedrooms > 0 else 0

    input_dict = {
        'property_type': prop_type.lower(),
        'city': city.lower(),
        'location_clean': location.lower(),
        'baths': baths,
        'bedrooms': bedrooms,
        'area_marla': area_marla,
        'total_rooms': bedrooms + baths,
        'baths_per_bed': bpb,
        'distance_to_city_center': 0.0,
        'has_agency': 0,
        'listing_age_days': 0,
        'society_avg_ppm': soc_avg,
    }

    features = assets['features']
    cat_cols = assets['cat_cols']
    input_df = pd.DataFrame([input_dict])[features]

    # XGBoost prediction
    xgb_df = input_df.copy()
    for c in cat_cols:
        xgb_df[c] = xgb_df[c].astype('category')

    pred_log = assets['model'].predict(xgb_df)[0]
    pred_price = float(np.expm1(pred_log))

    # Quantile predictions
    skl_df = input_df.copy()
    skl_df[cat_cols] = assets['encoder'].transform(skl_df[cat_cols])
    lo = float(np.expm1(assets['q_lower'].predict(skl_df)[0]))
    hi = float(np.expm1(assets['q_upper'].predict(skl_df)[0]))
    lo, hi = min(lo, pred_price), max(hi, pred_price)

    # Display
    st.success(f"### Estimated Price: {format_pkr(pred_price)}")
    st.info(f"**Market Range (10th–90th Percentile):** {format_pkr(lo)} — {format_pkr(hi)}")

    # SHAP
    st.subheader("📊 Price Drivers (SHAP in PKR)")
    st.markdown("Shows how each feature pushed the price up ↑ or down ↓ from the average property price.")
    if assets.get('shap_explainer') is not None:
        try:
            plt.style.use('dark_background')
            shap_vals = assets['shap_explainer'](xgb_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP error: {e}")
    else:
        st.info("SHAP explainability is currently unavailable in this environment.")

# ============================================================================
# EDA GALLERY
# ============================================================================
st.markdown("---")
st.subheader("📈 Data Insights")

report_files = sorted([f for f in os.listdir('reports') if f.endswith('.png')]) if os.path.exists('reports') else []
if report_files:
    cols = st.columns(2)
    for i, fname in enumerate(report_files):
        with cols[i % 2]:
            st.image(f'reports/{fname}', use_container_width=True)
else:
    st.info("Run `python eda.py` to generate data insight charts.")

st.markdown("---")
st.markdown("**Trained on 1.4 Lakh+ real Zameen.com listings** | Model: XGBoost + Quantile Regression")
