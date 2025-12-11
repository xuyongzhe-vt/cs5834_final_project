import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from urban_project.analyser.analyse_poi import plot_category_drivers
from urban_project.analyser.analyse_distance import plot_distance_effect
from urban_project.analyser.analyse_season import plot_seasonal_rhythms
import os

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "output"  # Directory to save figures


def analyse_and_draw_plots(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_category_drivers(df, OUTPUT_DIR)
    plot_distance_effect(df, OUTPUT_DIR)
    plot_seasonal_rhythms(df, OUTPUT_DIR)
