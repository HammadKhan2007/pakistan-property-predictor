"""
train_model.py â€” Advanced Model Comparison Pipeline
Trains Baseline â†’ Ridge â†’ RandomForest â†’ XGBoost, evaluates each,
then trains quantile models for prediction intervals and a raw-price
SHAP explainer. Logs every experiment to JSON for reproducibility.
"""
import pandas as pd
import numpy as np
import os, json, datetime, argparse, warnings
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score, median_absolute_error)
from sklearn.preprocessing import OrdinalEncoder
import joblib, shap

from data_pipeline import clean_data, feature_engineering

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
FEATURES = [
    'property_type', 'city', 'location_clean',
    'baths', 'bedrooms', 'area_marla',
    'total_rooms', 'baths_per_bed',
    'distance_to_city_center', 'has_agency', 'listing_age_days',
]
CAT_COLS = ['property_type', 'city', 'location_clean']

# ============================================================================
# HELPERS
# ============================================================================

def safe_mape(y_true, y_pred):
    """MAPE clipped to avoid inf from zero-price rows."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    print(f" MAE          : PKR {mae:>15,.0f}")
    print(f" Median AE    : PKR {mdae:>15,.0f}")
    print(f" RMSE         : PKR {rmse:>15,.0f}")
    print(f" MAPE         : {mape:>14.2f}%")
    print(f" RÂ² Score     : {r2:>14.4f}")
    return {'mae': mae, 'median_ae': mdae, 'rmse': rmse, 'mape': mape, 'r2': r2}

def log_experiment(metrics_dict, dataset_path, n_samples, features_used, model_name):
    """Append experiment results to a JSON log file."""
    log_path = 'models/experiment_log.json'
    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model_name,
        'dataset': dataset_path,
        'n_samples': n_samples,
        'n_features': len(features_used),
        'features': features_used,
        'metrics': {k: round(v, 4) for k, v in metrics_dict.items()},
    }
    logs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = json.load(f)
    logs.append(entry)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

# ============================================================================
# MAIN
# ============================================================================

def train(df_path):
    print("=" * 60)
    print(" PAKISTAN PROPERTY PRICE PREDICTOR â€” TRAINING PIPELINE")
    print("=" * 60)

    # 1. LOAD & CLEAN
    print("\n[1/7] Loading & cleaning data ...")
    df = pd.read_csv(df_path)
    df = clean_data(df)
    df = feature_engineering(df)
    print(f"      Clean dataset: {len(df):,} rows Ã— {len(df.columns)} cols")

    # 2. PREPARE FEATURES
    print("\n[2/7] Preparing features ...")
    X = df[FEATURES].copy()
    y_log = df['log_price']
    y_raw = df['price']

    for c in CAT_COLS:
        X[c] = X[c].astype('category')

    # 3. TRAIN-TEST SPLIT (then compute society avg on train only)
    print("\n[3/7] Splitting data (80/20) ...")
    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = \
        train_test_split(X, y_log, y_raw, test_size=0.2, random_state=42)

    # Leak-free society average
    train_tmp = X_train.copy()
    train_tmp['ppm'] = y_raw_train.values / train_tmp['area_marla'].values
    society_avg_map = train_tmp.groupby(['city', 'location_clean'])['ppm'].median().to_dict()
    global_median = train_tmp['ppm'].median()

    def map_soc(row):
        return society_avg_map.get((row['city'], row['location_clean']), global_median)

    X_train['society_avg_ppm'] = X_train.apply(map_soc, axis=1)
    X_test['society_avg_ppm']  = X_test.apply(map_soc, axis=1)

    features_final = FEATURES + ['society_avg_ppm']
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"      Features ({len(features_final)}): {features_final}")

    # 4. ENCODE FOR SKLEARN MODELS
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_enc = X_train.copy()
    X_train_enc[CAT_COLS] = enc.fit_transform(X_train[CAT_COLS])
    X_test_enc = X_test.copy()
    X_test_enc[CAT_COLS] = enc.transform(X_test[CAT_COLS])

    # 5. MODEL COMPARISON
    print("\n[4/7] Training & comparing models ...")

    # --- Baseline: predict median ---
    median_price = y_raw_train.median()
    preds_baseline = np.full(len(y_raw_test), median_price)
    m_base = evaluate(y_raw_test.values, preds_baseline, "Baseline (Global Median)")
    log_experiment(m_base, df_path, len(df), ['median'], 'baseline_median')

    # --- Ridge Regression ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_enc, y_log_train)
    preds_ridge = np.expm1(ridge.predict(X_test_enc))
    preds_ridge = np.clip(preds_ridge, 0, None)
    m_ridge = evaluate(y_raw_test.values, preds_ridge, "Ridge Regression (Log Target)")
    log_experiment(m_ridge, df_path, len(df), features_final, 'ridge')

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train_enc, y_log_train)
    preds_rf = np.expm1(rf.predict(X_test_enc))
    preds_rf = np.clip(preds_rf, 0, None)
    m_rf = evaluate(y_raw_test.values, preds_rf, "Random Forest (Log Target)")
    log_experiment(m_rf, df_path, len(df), features_final, 'random_forest')

    # --- XGBoost (Main Model) ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        random_state=42, enable_categorical=True, tree_method='hist',
    )
    xgb_model.fit(X_train, y_log_train,
                  eval_set=[(X_test, y_log_test)], verbose=100)
    preds_xgb = np.expm1(xgb_model.predict(X_test))
    preds_xgb = np.clip(preds_xgb, 0, None)
    m_xgb = evaluate(y_raw_test.values, preds_xgb, "XGBoost (Log Target) â˜… BEST")
    log_experiment(m_xgb, df_path, len(df), features_final, 'xgboost')

    # 6. GEOGRAPHIC CROSS-VALIDATION
    print("\n[5/7] Geographic Cross-Validation (GroupKFold on location) ...")
    gkf = GroupKFold(n_splits=5)
    geo_scores = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train_enc, y_log_train,
                                                        groups=X_train['location_clean']), 1):
        _rf = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
        _rf.fit(X_train_enc.iloc[tr_idx], y_log_train.iloc[tr_idx])
        _p = np.expm1(_rf.predict(X_train_enc.iloc[te_idx]))
        _p = np.clip(_p, 0, None)
        _mae = mean_absolute_error(np.expm1(y_log_train.iloc[te_idx]), _p)
        geo_scores.append(_mae)
        print(f"      Fold {fold}: MAE = PKR {_mae:,.0f}")
    print(f"      Avg Geographic CV MAE: PKR {np.mean(geo_scores):,.0f}")

    # 7. QUANTILE MODELS + SHAP + SAVE
    print("\n[6/7] Training quantile models for prediction intervals ...")
    q_lower = GradientBoostingRegressor(loss='quantile', alpha=0.10,
                                        n_estimators=100, max_depth=5, random_state=42)
    q_upper = GradientBoostingRegressor(loss='quantile', alpha=0.90,
                                        n_estimators=100, max_depth=5, random_state=42)
    q_lower.fit(X_train_enc, y_log_train)
    q_upper.fit(X_train_enc, y_log_train)

    # Interval coverage check
    lo = np.expm1(q_lower.predict(X_test_enc))
    hi = np.expm1(q_upper.predict(X_test_enc))
    covered = np.sum((y_raw_test.values >= lo) & (y_raw_test.values <= hi))
    coverage = covered / len(y_raw_test) * 100
    print(f"      80% Prediction Interval Coverage: {coverage:.1f}%")

    print("\n[7/7] Saving models & artifacts ...")
    os.makedirs('models', exist_ok=True)
    xgb_model.save_model('models/xgb_model.json')
    joblib.dump(q_lower, 'models/lgb_lower.pkl')
    joblib.dump(q_upper, 'models/lgb_upper.pkl')
    joblib.dump(enc, 'models/cat_encoder.pkl')
    joblib.dump(features_final, 'models/features.pkl')
    joblib.dump(CAT_COLS, 'models/cat_cols.pkl')

    # Locations & society map
    cities = df['city'].unique().tolist()
    locs = {c: df[df['city'] == c]['location_clean'].unique().tolist() for c in cities}
    joblib.dump({
        'cities': cities, 'locations': locs,
        'society_avg_map': society_avg_map, 'global_median': global_median,
    }, 'models/locations.pkl')

    # SHAP on raw price
    print("      Training SHAP explainer (raw PKR) ...")
    shap_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        random_state=42, enable_categorical=True, tree_method='hist',
    )
    shap_model.fit(X_train, y_raw_train)
    explainer = shap.TreeExplainer(shap_model)
    joblib.dump(explainer, 'models/shap_explainer_raw.pkl')

    # Underpriced listings
    test_out = X_test.copy()
    test_out['actual_price'] = y_raw_test.values
    test_out['predicted_price'] = preds_xgb
    test_out['deal_score'] = (test_out['predicted_price'] / test_out['actual_price'] - 1) * 100
    underpriced = test_out[test_out['deal_score'] > 30].sort_values('deal_score', ascending=False)
    underpriced.to_csv('data/underpriced_opportunities.csv', index=False)
    print(f"      Found {len(underpriced):,} underpriced listings â†’ data/underpriced_opportunities.csv")

    print("\nâœ… Training complete! All models saved to models/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/Property_with_Feature_Engineering.csv')
    args = p.parse_args()
    if os.path.exists(args.data):
        train(args.data)
    else:
        print(f"âŒ Data file not found: {args.data}")
