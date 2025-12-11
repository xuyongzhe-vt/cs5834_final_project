import os
import seaborn as sns
import matplotlib.pyplot as plt


def get_top_categories(df, n=10):
    """Returns the list of top N categories by row count."""
    return df['TOP_CATEGORY'].value_counts().head(n).index.tolist()


# ==========================================
# FIGURE 1: CATEGORY ANALYSIS (The Baseline)
# ==========================================
def plot_category_drivers(df, output_dir):
    print("Generating Figure 1: Category Drivers...")
    top_cats = get_top_categories(df, n=10)
    df_filtered = df[df['TOP_CATEGORY'].isin(top_cats)].copy()

    # Sort orders for clean visualization
    order_visits = df_filtered.groupby('TOP_CATEGORY')['log_visits'].median().sort_values(ascending=False).index
    order_dwell = df_filtered.groupby('TOP_CATEGORY')['MEDIAN_DWELL'].median().sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Subplot 1: Visit Intensity (Volume)
    sns.boxplot(data=df_filtered, x='log_visits', y='TOP_CATEGORY',
                order=order_visits, ax=axes[0], palette="Blues_r", showfliers=False)
    axes[0].set_title('Volume Driver: Visit Intensity by Category', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Log(Visits)', fontsize=12)
    axes[0].set_ylabel('')

    # Subplot 2: Stickiness (Quality)
    sns.boxplot(data=df_filtered, x='MEDIAN_DWELL', y='TOP_CATEGORY',
                order=order_dwell, ax=axes[1], palette="Oranges_r", showfliers=False)
    axes[1].set_title('Quality Driver: Stickiness (Dwell Time) by Category', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Median Dwell Time (Minutes)', fontsize=12)
    axes[1].set_ylabel('')

    # Limit x-axis for dwell if there are extreme outliers (e.g., hotels)
    # axes[1].set_xlim(0, 200)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_category_drivers.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()