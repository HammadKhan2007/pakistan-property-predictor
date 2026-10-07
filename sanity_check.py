"""
sanity_check.py — Model Validation & LinkedIn Insights Generator
Tests the trained model on a holdout sample and produces per-city
breakdowns plus prediction interval coverage statistics.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib, warnings
from sklearn.metrics import median_absolute_error
from data_pipeline import clean_data, feature_engineering

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print(" MODEL VALIDATION & SANITY CHECK")
    print("=" * 60)

    # Load models
    model = xgb.XGBRegressor()
    model.load_model('models/xgb_model.json')
    q_lo = joblib.load('models/lgb_lower.pkl')
    q_hi = joblib.load('models/lgb_upper.pkl')
    features = joblib.load('models/features.pkl')
    cat_cols = joblib.load('models/cat_cols.pkl')
    enc = joblib.load('models/cat_encoder.pkl')
    loc_data = joblib.load('models/locations.pkl')
    soc_map = loc_data['society_avg_map']
    g_med = loc_data['global_median']

    # Load & clean fresh data
    df = pd.read_csv('data/Property_with_Feature_Engineering.csv')
    df = clean_data(df)
    df = feature_engineering(df)

    # Random 50 listings
    sample = df.sample(50, random_state=101)
    X = sample[features[:-1]].copy()  # exclude society_avg_ppm
    y = sample['price'].values

    # Add society avg from training map
    def map_soc(row):
        return soc_map.get((row['city'], row['location_clean']), g_med)
    X['society_avg_ppm'] = X.apply(map_soc, axis=1)
    X = X[features]

    # XGBoost
    X_xgb = X.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].astype('category')

    # Sklearn
    X_skl = X.copy()
    X_skl[cat_cols] = enc.transform(X_skl[cat_cols])

    preds = np.clip(np.expm1(model.predict(X_xgb)), 0, None)
    lo = np.clip(np.expm1(q_lo.predict(X_skl)), 0, None)
    hi = np.clip(np.expm1(q_hi.predict(X_skl)), 0, None)

    # Overall metrics
    mdae = median_absolute_error(y, preds)
    mask = y > 0
    mape = np.median(np.abs((y[mask] - preds[mask]) / y[mask])) * 100
    covered = np.sum((y >= lo) & (y <= hi))
    coverage = covered / len(y) * 100

    print(f"\nTested on {len(y)} holdout listings")
    print(f"  Median Absolute Error : PKR {mdae:>15,.0f}")
    print(f"  Median % Error        : {mape:>14.1f}%")
    print(f"  80% Interval Coverage : {coverage:>14.1f}%")

    # Per-city breakdown
    print(f"\n{'City':<15} {'Median AE (PKR)':>20} {'Median %Err':>15} {'Listings':>10}")
    print("-" * 62)
    sample_eval = sample.copy()
    sample_eval['pred'] = preds
    for city in sorted(sample_eval['city'].unique()):
        sub = sample_eval[sample_eval['city'] == city]
        _y = sub['price'].values
        _p = sub['pred'].values
        _mdae = median_absolute_error(_y, _p)
        _mask = _y > 0
        _mape = np.median(np.abs((_y[_mask] - _p[_mask]) / _y[_mask])) * 100
        print(f"{city.title():<15} {_mdae:>20,.0f} {_mape:>14.1f}% {len(sub):>10}")

    print(f"\n{'='*60}")
    print(f" LINKEDIN POST HOOK:")
    print(f" \"I built a model that predicts Pakistani property prices")
    print(f"  within {mape:.0f}% median error, and its 80% prediction")
    print(f"  interval covers {coverage:.0f}% of actual prices.\"")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
