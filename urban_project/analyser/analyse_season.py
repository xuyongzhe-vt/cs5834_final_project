import os
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================
# FIGURE 3: TIME ANALYSIS (The Rhythm)
# ==========================================
def plot_seasonal_rhythms(df, output_dir):
    print("Generating Figure 3: Seasonal Rhythms...")

    # 1. Define distinct categories for the heatmap (8 Categories)
    # Dictionary format: {Display Name: [List of keywords to match]}
    categories_of_interest = {
        'Education': ['School', 'University', 'College', 'Education'],
        'Recreation (Parks)': ['Park', 'Nature', 'Camp', 'Recreation'],
        'Fitness (Gyms)': ['Gym', 'Fitness', 'Sports', 'Club'],
        'Dining': ['Restaurant', 'Eating', 'Bar', 'Dining'],
        'Shopping (Malls)': ['Mall', 'Shopping', 'Outlet'],
        'Groceries': ['Grocery', 'Supermarket', 'Food Store'],
        'Lodging (Hotels)': ['Hotel', 'Motel', 'Accommodation'],
        'Health': ['Physician', 'Dentist', 'Doctor', 'Hospital', 'Clinic']
    }

    def get_display_label(cat):
        s = str(cat).lower()
        for label, keywords in categories_of_interest.items():
            if any(k.lower() in s for k in keywords):
                return label
        return None

    # Filter df to just rows that match one of our categories
    df_seasonal = df.copy()
    df_seasonal['Display_Cat'] = df_seasonal['TOP_CATEGORY'].apply(get_display_label)
    df_seasonal = df_seasonal.dropna(subset=['Display_Cat'])

    # Print found categories for verification
    print(f"Heatmap Categories Found: {sorted(df_seasonal['Display_Cat'].unique())}")

    # 2. Aggregation: Group by Category and Month
    # Volume (Sum of visits)
    pivot_vol = df_seasonal.groupby(['Display_Cat', 'Month'])['RAW_VISIT_COUNTS'].sum().reset_index()
    # Quality (Mean of Median Dwell)
    pivot_dwell = df_seasonal.groupby(['Display_Cat', 'Month'])['MEDIAN_DWELL'].mean().reset_index()

    # 3. Pivot for Heatmap (Rows=Category, Cols=Month)
    heatmap_vol = pivot_vol.pivot(index='Display_Cat', columns='Month', values='RAW_VISIT_COUNTS')
    heatmap_dwell = pivot_dwell.pivot(index='Display_Cat', columns='Month', values='MEDIAN_DWELL')

    # 4. Normalization (Min-Max) for Volume so we see the *shape* not just magnitude
    heatmap_vol_norm = heatmap_vol.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    # 5. Plotting Dual Heatmap
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: Volume Rhythm
    sns.heatmap(heatmap_vol_norm, ax=axes[0], cmap="Blues", annot=False, linewidths=.5)
    axes[0].set_title('Rhythm of Quantity: Normalized Visit Volume', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('')

    # Bottom: Quality Rhythm
    # CHANGED: 'Magma_r' -> 'magma_r' to fix KeyError
    sns.heatmap(heatmap_dwell, ax=axes[1], cmap="magma_r", annot=True, fmt=".0f", linewidths=.5)
    axes[1].set_title('Rhythm of Quality: Median Dwell Time (Minutes)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    axes[1].set_xlabel('Month', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig3_seasonal_rhythms.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()
