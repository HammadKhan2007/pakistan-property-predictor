"""
eda.py â€” Exploratory Data Analysis Script
Generates publication-quality charts saved to reports/ folder.
Run: python eda.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_pipeline import clean_data, feature_engineering

plt.style.use('dark_background')
os.makedirs('reports', exist_ok=True)

print("Loading and cleaning data ...")
df = pd.read_csv('data/Property_with_Feature_Engineering.csv')
df = clean_data(df)
df = feature_engineering(df)
print(f"Clean dataset: {len(df):,} rows")

# ============================================================================
# 1. Price Distribution (Log Scale)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['price'] / 1e7, bins=100, color='#00d4aa', edgecolor='none', alpha=0.8)
axes[0].set_xlabel('Price (Crore PKR)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution (Raw)')
axes[0].set_xlim(0, 20)

axes[1].hist(df['log_price'], bins=100, color='#ff6b6b', edgecolor='none', alpha=0.8)
axes[1].set_xlabel('Log(Price)')
axes[1].set_ylabel('Count')
axes[1].set_title('Price Distribution (Log Transformed)')

plt.suptitle('Why We Use Log Transform â€” Prices Are Heavily Right-Skewed', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/01_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Done: 01_price_distribution.png")

# ============================================================================
# 2. Price by City (Box Plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
city_order = df.groupby('city')['price'].median().sort_values(ascending=False).index
data_by_city = [df[df['city'] == c]['price'].values / 1e7 for c in city_order]
bp = ax.boxplot(data_by_city, tick_labels=[c.title() for c in city_order],
                patch_artist=True, showfliers=False,
                medianprops=dict(color='white', linewidth=2))
colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Price (Crore PKR)')
ax.set_title('Property Prices by City', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/02_price_by_city.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 02_price_by_city.png")

# ============================================================================
# 3. Top 15 Most Expensive Societies (Median Price per Marla)
# ============================================================================
society_stats = df.groupby(['city', 'location_clean']).agg(
    median_ppm=('price_per_marla', 'median'),
    count=('price', 'count')
).reset_index()
society_stats = society_stats[society_stats['count'] >= 20]  # At least 20 listings
top15 = society_stats.nlargest(15, 'median_ppm')

fig, ax = plt.subplots(figsize=(12, 7))
labels = [f"{r['location_clean'].title()} ({r['city'].title()})" for _, r in top15.iterrows()]
bars = ax.barh(labels[::-1], (top15['median_ppm'].values / 1e6)[::-1], color='#00d4aa', alpha=0.85)
ax.set_xlabel('Median Price per Marla (Million PKR)')
ax.set_title('Top 15 Most Expensive Societies in Pakistan', fontsize=14, fontweight='bold')
for bar, val in zip(bars, (top15['median_ppm'].values / 1e6)[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}M', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('reports/03_top15_expensive_societies.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 03_top15_expensive_societies.png")

# ============================================================================
# 4. Area vs Price (Scatter + Property Type)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))
for ptype in ['house', 'flat', 'upper portion', 'lower portion']:
    subset = df[df['property_type'] == ptype].sample(min(2000, len(df[df['property_type'] == ptype])), random_state=42)
    ax.scatter(subset['area_marla'], subset['price'] / 1e7,
               alpha=0.3, s=10, label=ptype.title())
ax.set_xlabel('Area (Marla)')
ax.set_ylabel('Price (Crore PKR)')
ax.set_title('Area vs Price by Property Type', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.legend()
plt.tight_layout()
plt.savefig('reports/04_area_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 04_area_vs_price.png")

# ============================================================================
# 5. Correlation Heatmap
# ============================================================================
numeric_cols = ['price', 'area_marla', 'bedrooms', 'baths', 'total_rooms',
                'baths_per_bed', 'distance_to_city_center', 'has_agency']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([c.replace('_', '\n') for c in numeric_cols], fontsize=8, rotation=45, ha='right')
ax.set_yticklabels([c.replace('_', '\n') for c in numeric_cols], fontsize=8)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                fontsize=8, color='black' if abs(corr.iloc[i, j]) < 0.5 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 05_correlation_heatmap.png")

# ============================================================================
# 6. Distance to City Center vs Price
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
for city_name in ['lahore', 'karachi', 'islamabad']:
    subset = df[df['city'] == city_name].sample(min(2000, len(df[df['city'] == city_name])), random_state=42)
    ax.scatter(subset['distance_to_city_center'], subset['price'] / 1e7,
               alpha=0.25, s=10, label=city_name.title())
ax.set_xlabel('Distance to City Center (km)')
ax.set_ylabel('Price (Crore PKR)')
ax.set_title('Properties Closer to City Center Are More Expensive', fontsize=14, fontweight='bold')
ax.set_xlim(0, 50)
ax.set_ylim(0, 30)
ax.legend()
plt.tight_layout()
plt.savefig('reports/06_distance_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 06_distance_vs_price.png")

# ============================================================================
# 7. Property Type Distribution
# ============================================================================
type_counts = df['property_type'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(type_counts.index.str.title(), type_counts.values,
              color=['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6', '#e74c3c', '#1abc9c'],
              alpha=0.85)
for bar, val in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}', ha='center', fontsize=9)
ax.set_ylabel('Number of Listings')
ax.set_title('Property Type Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('reports/07_property_type_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("âœ… 07_property_type_dist.png")

print(f"\nðŸŽ‰ All EDA charts saved to reports/ folder!")
