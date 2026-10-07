import os
import joblib
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Pakistan Property Price Predictor API", version="1.0")

MODEL_DIR = "models"
MODEL_VERSION = "v1.0"

# Lazy load models and artifacts
_xgb_model = None
_lgb_lower = None
_lgb_upper = None
_locations_data = None
_cat_encoder = None
_features_list = None
_cat_cols = None

def get_xgb_model():
    global _xgb_model
    if _xgb_model is None:
        _xgb_model = xgb.XGBRegressor()
        model_path = os.path.join(MODEL_DIR, "xgb_model.json")
        if os.path.exists(model_path):
            _xgb_model.load_model(model_path)
    return _xgb_model

def get_lgb_lower():
    global _lgb_lower
    if _lgb_lower is None:
        model_path = os.path.join(MODEL_DIR, "lgb_lower.pkl")
        if os.path.exists(model_path):
            _lgb_lower = joblib.load(model_path)
    return _lgb_lower

def get_lgb_upper():
    global _lgb_upper
    if _lgb_upper is None:
        model_path = os.path.join(MODEL_DIR, "lgb_upper.pkl")
        if os.path.exists(model_path):
            _lgb_upper = joblib.load(model_path)
    return _lgb_upper

def get_locations_data():
    global _locations_data
    if _locations_data is None:
        path = os.path.join(MODEL_DIR, "locations.pkl")
        if os.path.exists(path):
            _locations_data = joblib.load(path)
    return _locations_data

def get_cat_encoder():
    global _cat_encoder
    if _cat_encoder is None:
        path = os.path.join(MODEL_DIR, "cat_encoder.pkl")
        if os.path.exists(path):
            _cat_encoder = joblib.load(path)
    return _cat_encoder

def get_features_list():
    global _features_list
    if _features_list is None:
        path = os.path.join(MODEL_DIR, "features.pkl")
        if os.path.exists(path):
            _features_list = joblib.load(path)
    return _features_list

def get_cat_cols():
    global _cat_cols
    if _cat_cols is None:
        path = os.path.join(MODEL_DIR, "cat_cols.pkl")
        if os.path.exists(path):
            _cat_cols = joblib.load(path)
    return _cat_cols

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    city: str
    location: str
    property_type: str
    area_value: float
    area_unit: str
    bedrooms: int
    bathrooms: int

class PredictionResponse(BaseModel):
    estimated_price: float
    estimated_price_formatted: str
    price_range_lower: float
    price_range_upper: float
    price_range_formatted: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_version: str

# --- Helper functions ---

def haversine(lat1, lon1, lat2, lon2):
    # Radius of earth in km
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

CITY_CENTERS = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4504, 73.1350)
}

def format_price(amount):
    if amount >= 1e7:
        return f"PKR {amount/1e7:.2f} Crore"
    elif amount >= 1e5:
        return f"PKR {amount/1e5:.2f} Lakh"
    else:
        return f"PKR {amount:,.0f}"

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    # Preload models
    get_xgb_model()
    get_lgb_lower()
    get_lgb_upper()
    get_locations_data()
    get_cat_encoder()
    get_features_list()
    get_cat_cols()

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.get("/locations")
def get_locations():
    loc_data = get_locations_data()
    if not loc_data:
        raise HTTPException(status_code=500, detail="Locations data not loaded.")
    cities = loc_data.get("cities", [])
    locations_map = loc_data.get("locations", {})
    return {"cities": cities, "locations": locations_map}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Load requirements
    xgb_model = get_xgb_model()
    lgb_lower = get_lgb_lower()
    lgb_upper = get_lgb_upper()
    loc_data = get_locations_data()
    cat_encoder = get_cat_encoder()
    features = get_features_list()
    cat_cols = get_cat_cols()

    if None in [xgb_model, lgb_lower, lgb_upper, loc_data, cat_encoder, features, cat_cols]:
        raise HTTPException(status_code=500, detail="Model artifacts are not fully loaded.")

    # 1. Feature Engineering
    # Area conversion
    area_marla = req.area_value
    if req.area_unit.lower() == "kanal":
        area_marla *= 20
    elif req.area_unit.lower() == "sq. yd.":
        area_marla *= (1/30.25)
    elif req.area_unit.lower() == "sq. ft.":
        area_marla *= (1/272.25)
    elif req.area_unit.lower() == "sq. m.":
        area_marla *= (1/25.29)
    
    baths_per_bed = req.bathrooms / req.bedrooms if req.bedrooms > 0 else 0
    total_rooms = req.bedrooms + req.bathrooms
    
    dist = 0
    if req.city in CITY_CENTERS:
        lat1, lon1 = CITY_CENTERS[req.city]
        lat2, lon2 = lat1, lon1 # Simple approximation
        dist = haversine(lat1, lon1, lat2, lon2)

    has_agency = 0
    listing_age_days = 0
    
    loc_key = f"{req.city}_{req.location}"
    society_avg_map = loc_data.get("society_avg_map", {})
    global_median = loc_data.get("global_median", 0)
    society_avg_price = society_avg_map.get(loc_key, global_median)

    # 2. Build feature dictionary
    feat_dict = {
        "city": req.city,
        "location": req.location,
        "property_type": req.property_type,
        "baths": req.bathrooms,
        "bedrooms": req.bedrooms,
        "Area Marla": area_marla,
        "has_agency": has_agency,
        "listing_age_days": listing_age_days,
        "baths_per_bed": baths_per_bed,
        "total_rooms": total_rooms,
        "distance_to_city_center": dist,
        "society_avg_price": society_avg_price
    }

    df = pd.DataFrame([feat_dict])
    
    # Fill missing features with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    df = df[features] # Order columns

    # 3. Predict XGBoost
    # XGBoost requires category types
    df_xgb = df.copy()
    for col in cat_cols:
        df_xgb[col] = df_xgb[col].astype("category")
    
    # Predict
    dmatrix = xgb.DMatrix(df_xgb, enable_categorical=True)
    log_pred = xgb_model.predict(dmatrix)[0]
    estimated_price = float(np.expm1(log_pred))

    # 4. Predict Quantiles
    df_sklearn = df.copy()
    df_sklearn[cat_cols] = cat_encoder.transform(df_sklearn[cat_cols])
    
    log_lower = lgb_lower.predict(df_sklearn)[0]
    log_upper = lgb_upper.predict(df_sklearn)[0]
    
    price_lower = float(np.expm1(log_lower))
    price_upper = float(np.expm1(log_upper))
    
    return PredictionResponse(
        estimated_price=estimated_price,
        estimated_price_formatted=format_price(estimated_price),
        price_range_lower=price_lower,
        price_range_upper=price_upper,
        price_range_formatted=f"{format_price(price_lower)} — {format_price(price_upper)}",
        model_version=MODEL_VERSION
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
