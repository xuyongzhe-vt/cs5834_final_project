import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def map_broad_sector(cat):
    """Helper to group specific NAICS categories into broad sectors for Fig 2."""
    cat = str(cat).lower()
    if 'restaurant' in cat or 'eating' in cat or 'food' in cat:
        return 'Food & Dining'
    elif 'retail' in cat or 'store' in cat or 'mall' in cat or 'shopping' in cat:
        return 'Retail'
    elif 'school' in cat or 'university' in cat or 'college' in cat:
        return 'Education'
    elif 'gym' in cat or 'fitness' in cat:
        return 'Fitness'
    elif 'park' in cat or 'nature' in cat:
        return 'Recreation'
    else:
        return 'Other'


# ==========================================
# FIGURE 2: DISTANCE ANALYSIS (The Relation)
# ==========================================
def plot_distance_effect(df, output_dir):
    print("Generating Figure 2: Distance Effect...")
    # 1. Filter valid distances
    df_clean = df.dropna(subset=['DISTANCE_FROM_HOME']).copy()
    df_clean = df_clean[df_clean['DISTANCE_FROM_HOME'] > 0]

    # 2. Convert meters to km
    df_clean['dist_km'] = df_clean['DISTANCE_FROM_HOME'] / 1000.0

    # 3. Create Distance Bins (Logic from Plan)
    # <5km (Community), 5-15km (City), 15-50km (Suburban), >50km (Travel)
    bins = [0, 5, 15, 50, 10000]
    labels = ['Community (<5km)', 'City (5-15km)', 'Regional (15-50km)', 'Travel (>50km)']
    df_clean['Dist_Bucket'] = pd.cut(df_clean['dist_km'], bins=bins, labels=labels)

    # 4. Broad Category Mapping (Food vs Retail)
    df_clean['Sector'] = df_clean['TOP_CATEGORY'].apply(map_broad_sector)
    # Filter to only relevant sectors for the story
    # MODIFIED: Expanded to include Recreation, Fitness, and Education
    target_sectors = ['Food & Dining', 'Retail', 'Recreation', 'Fitness', 'Education']
    df_sector = df_clean[df_clean['Sector'].isin(target_sectors)]

    # 5. Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sector, x='Dist_Bucket', y='MEDIAN_DWELL', hue='Sector',
                 marker='o', style='Sector', markersize=8, linewidth=2.5)

    plt.title('The "Sunk Cost" Effect: Distance vs. Dwell Time', fontsize=14, fontweight='bold')
    plt.xlabel('Distance from Home (Catchment Area)', fontsize=12)
    plt.ylabel('Average Dwell Time (Minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig2_distance_effect.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()
