import pandas as pd
import numpy as np
import os

def generate_mock_data(n_samples=5000):
    np.random.seed(42)
    
    cities = ['lahore', 'karachi', 'islamabad', 'rawalpindi']
    prop_types = ['house', 'plot', 'flat', 'lower portion', 'upper portion']
    
    # Base prices per Marla in different cities (roughly)
    base_price_marla = {
        'lahore': 2000000,
        'karachi': 2500000,
        'islamabad': 3000000,
        'rawalpindi': 1500000
    }
    
    # Mock locations per city
    locations = {
        'lahore': ['dha phase 6', 'dha phase 8', 'bahria town', 'wapda town', 'johar town', 'cantt'],
        'karachi': ['dha phase 8', 'clifton', 'gulshan-e-iqbal', 'bahria town karachi', 'north nazimabad'],
        'islamabad': ['sector f-8', 'sector e-7', 'dha islamabad', 'bahria town', 'sector g-13'],
        'rawalpindi': ['bahria town', 'pwd housing scheme', 'chaklala scheme', 'askari 14']
    }
    
    data = []
    
    for _ in range(n_samples):
        city = np.random.choice(cities)
        loc = np.random.choice(locations[city])
        prop_type = np.random.choice(prop_types, p=[0.5, 0.3, 0.1, 0.05, 0.05])
        
        # Area from 2 Marla to 40 Marla (2 Kanal)
        area_marla = np.random.lognormal(mean=np.log(10), sigma=0.8)
        area_marla = np.clip(area_marla, 2, 80)
        
        # Determine Area string format (sometimes Marla, sometimes Kanal, sometimes Sq. Yd.)
        format_choice = np.random.choice(['marla', 'kanal', 'sqyd'])
        if format_choice == 'marla':
            area_str = f"{area_marla:.1f} Marla"
        elif format_choice == 'kanal':
            area_str = f"{area_marla/20:.2f} Kanal"
        else:
            area_str = f"{area_marla*25:.0f} Sq. Yd."
            
        # Baths and Bedrooms
        if prop_type in ['house', 'flat', 'upper portion', 'lower portion']:
            bedrooms = int(max(1, area_marla / 2.5 + np.random.normal(0, 1)))
            baths = int(max(1, bedrooms + np.random.randint(-1, 2)))
        else:
            bedrooms = 0
            baths = 0
            
        # Price Calculation
        price = area_marla * base_price_marla[city]
        
        # Location multiplier
        if 'dha' in loc or 'sector e' in loc or 'sector f' in loc or 'clifton' in loc:
            price *= np.random.uniform(1.5, 2.5)
        elif 'bahria' in loc:
            price *= np.random.uniform(1.2, 1.8)
        else:
            price *= np.random.uniform(0.8, 1.2)
            
        # Property type multiplier
        if prop_type == 'house':
            price *= 1.5 # construction cost added
        elif prop_type == 'flat':
            price *= 1.2
        elif prop_type == 'upper portion' or prop_type == 'lower portion':
            price *= 0.6
            
        # Add some noise
        price *= np.random.uniform(0.9, 1.1)
        
        data.append({
            'property_id': np.random.randint(10000000, 99999999),
            'location': loc,
            'city': city,
            'property_type': prop_type,
            'price': int(price),
            'baths': baths,
            'bedrooms': bedrooms,
            'area': area_str,
            'purpose': 'For Sale'
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/zameen_sample.csv', index=False)
    print(f"Generated {n_samples} rows of mock data at data/zameen_sample.csv")

if __name__ == "__main__":
    generate_mock_data()
