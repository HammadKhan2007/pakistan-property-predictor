import pandas as pd
import numpy as np
import re
import difflib
import math

# ============================================================================
# UNIT CONVERSIONS & PARSERS
# ============================================================================

def parse_price(price_str):
    """
    Converts Pakistani price strings to numeric PKR.
    Examples: '1.5 Crore' -> 15000000, '85 Lakh' -> 8500000, '1.2 Arab' -> 1200000000
    """
    if pd.isna(price_str):
        return np.nan
    p = str(price_str).lower().replace(',', '').strip()
    num = re.search(r'([\d\.]+)', p)
    if not num:
        return np.nan
    val = float(num.group(1))
    if 'arab' in p:
        return val * 1e9
    elif 'crore' in p or 'cr' in p:
        return val * 1e7
    elif 'lakh' in p or 'lac' in p:
        return val * 1e5
    elif 'thousand' in p or 'k' in p:
        return val * 1e3
    return val

def parse_area(area_str):
    """
    Converts area strings to standardized Marla.
    Assumption: 1 Kanal = 20 Marla, 1 Marla ≈ 225 Sq. Ft., 1 Marla ≈ 25 Sq. Yd.
    """
    if pd.isna(area_str):
        return np.nan
    s = str(area_str).lower().replace(',', '')
    num = re.search(r'([\d\.]+)', s)
    if not num:
        return np.nan
    val = float(num.group(1))
    if 'kanal' in s:
        return val * 20.0
    elif 'marla' in s:
        return val
    elif 'sq. yd' in s or 'square yard' in s:
        return val / 25.0
    elif 'sq. ft' in s or 'square feet' in s:
        return val / 225.0
    elif 'sq. m' in s or 'square meter' in s:
        return val * 10.7639 / 225.0
    return val

# ============================================================================
# GEOSPATIAL
# ============================================================================

CITY_CENTERS = {
    'lahore':     (31.5204, 74.3587),
    'karachi':    (24.8607, 67.0011),
    'islamabad':  (33.6844, 73.0479),
    'rawalpindi': (33.5651, 73.0169),
    'faisalabad': (31.4504, 73.1350),
}

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def compute_city_center_distance(row):
    center = CITY_CENTERS.get(row['city'], None)
    if center and pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
        return haversine(row['latitude'], row['longitude'], center[0], center[1])
    return np.nan

# ============================================================================
# LOCATION FUZZY MATCHING
# ============================================================================

def standardize_locations(series):
    """
    Rule-based cleaning + difflib fuzzy matching to unify messy location strings.
    E.g. 'DHA Phase 5', 'dha ph-5', 'D.H.A Phase V' → 'dha phase 5'
    """
    unique = series.dropna().unique().tolist()
    
    # Step 1: Rule-based normalization
    rule_map = {}
    for loc in unique:
        c = loc.lower().replace('.', '').replace('-', ' ').strip()
        c = re.sub(r'\s+', ' ', c)
        # Roman numeral → digit
        for roman, digit in [('phase viii','phase 8'),('phase vii','phase 7'),
                             ('phase vi','phase 6'),('phase v','phase 5'),
                             ('phase iv','phase 4'),('phase iii','phase 3'),
                             ('phase ii','phase 2'),('phase i','phase 1')]:
            c = c.replace(roman, digit)
        c = c.replace('ph ', 'phase ')
        rule_map[loc] = c
    
    # Step 2: Fuzzy deduplicate
    final_map = {}
    canonical = []
    for raw, std in rule_map.items():
        matches = difflib.get_close_matches(std, canonical, n=1, cutoff=0.90)
        if matches:
            final_map[raw] = matches[0]
        else:
            canonical.append(std)
            final_map[raw] = std
    
    return series.map(final_map)

# ============================================================================
# CLEANING
# ============================================================================

def clean_data(df):
    """Full cleaning pipeline for Zameen.com data."""
    df = df.copy()
    
    # Parse price
    if df['price'].dtype == object:
        df['price'] = df['price'].apply(parse_price)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Parse area
    if 'area_marla' not in df.columns or df['area_marla'].isna().all():
        df['area_marla'] = df['area'].apply(parse_area)
    df['area_marla'] = pd.to_numeric(df['area_marla'], errors='coerce')
    
    # Filter to "For Sale" only
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].str.lower().str.strip()
        df = df[df['purpose'] == 'for sale']
    
    # Drop essential nulls
    df = df.dropna(subset=['price', 'area_marla', 'location', 'city'])
    
    # Outlier removal
    df['price_per_marla'] = df['price'] / df['area_marla']
    df = df[(df['area_marla'] >= 1) & (df['area_marla'] <= 2000)]
    df = df[(df['price'] > 100000) & (df['price_per_marla'] < 1.5e8)]
    
    # Clean strings
    df['city'] = df['city'].str.lower().str.strip()
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df['location_clean'] = standardize_locations(df['location'])
    
    # Numerics
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce').fillna(0).astype(int)
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0).astype(int)
    
    # Parse dates
    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['listing_year'] = df['date_added'].dt.year
        df['listing_month'] = df['date_added'].dt.month
        max_date = df['date_added'].max()
        df['listing_age_days'] = (max_date - df['date_added']).dt.days.fillna(0).astype(int)
    
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """Compute advanced features for modelling."""
    df = df.copy()
    
    # Room features
    df['total_rooms'] = df['bedrooms'] + df['baths']
    df['baths_per_bed'] = np.where(df['bedrooms'] > 0, df['baths'] / df['bedrooms'], 0)
    
    # Geospatial
    df['distance_to_city_center'] = df.apply(compute_city_center_distance, axis=1)
    df['distance_to_city_center'] = df['distance_to_city_center'].fillna(
        df.groupby('city')['distance_to_city_center'].transform('median')
    ).fillna(0)
    
    # Agency flag
    if 'agency' in df.columns:
        df['has_agency'] = df['agency'].notna().astype(int)
    else:
        df['has_agency'] = 0
    
    # Listing age
    if 'listing_age_days' not in df.columns:
        df['listing_age_days'] = 0
    
    # Log price
    df['log_price'] = np.log1p(df['price'])
    
    return df
