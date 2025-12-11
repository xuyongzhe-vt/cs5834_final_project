import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = 'input'  # The folder where your csv.gz files are
OUTPUT_FILE = 'processed_urban_data.parquet'  # The fast cache file

# The "Golden Columns" we selected in the plan
# We ONLY load these to save massive amounts of RAM
COLS_TO_USE = [
    'PLACEKEY',
    'TOP_CATEGORY',
    'LATITUDE',
    'LONGITUDE',
    'DATE_RANGE_START',
    'RAW_VISIT_COUNTS',
    'MEDIAN_DWELL',
    'DISTANCE_FROM_HOME',
    "CITY",
    "REGION",
    "POSTAL_CODE",
    "OPEN_HOURS",
    "ENCLOSED",
    "WKT_AREA_SQ_METERS",
    "DEVICE_TYPE"
]


def load_and_process_data(input_dir=INPUT_DIR):
    """
    Smart Loader: Checks for a cached parquet file first.
    If not found, loads all monthly CSVs, filters columns, adds Month column,
    concatenates, saves the cache, and returns the dataframe.
    """

    # --- 1. Fast Path: Check for Cache ---
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing cache: {OUTPUT_FILE}")
        print("Loading from Parquet (Fast)...")
        try:
            df = pd.read_parquet(OUTPUT_FILE)
            print("Cache loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading cache: {e}. Falling back to raw processing.")

    # --- 2. Slow Path: Load from Raw CSVs ---
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.csv.gz")))

    if not all_files:
        print(f"No files found in {input_dir}")
        return None

    print(f"Found {len(all_files)} files. Starting load...")

    df_list = []

    for filename in all_files:
        print(f"Loading {os.path.basename(filename)}...")
        try:
            # Extract month from filename (e.g., "2021-01.csv.gz" -> 1)
            basename = os.path.basename(filename)
            month_str = basename.split('.')[0].split('-')[1]
            month_int = int(month_str)

            # Load specific columns
            df_temp = pd.read_csv(
                filename,
                compression='gzip',
                usecols=COLS_TO_USE
            )

            # Add Month column
            df_temp['Month'] = month_int

            # Basic Cleaning
            subset_cols = ['PLACEKEY', 'LATITUDE', 'LONGITUDE', 'RAW_VISIT_COUNTS']
            subset_cols = [c for c in subset_cols if c in df_temp.columns]
            df_temp = df_temp.dropna(subset=subset_cols)

            df_list.append(df_temp)

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if not df_list:
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Merged Data Shape: {full_df.shape}")

    # ==========================================
    # DATA CLEANING & TARGET PREP
    # ==========================================
    print("Finalizing Data Types...")

    if 'RAW_VISIT_COUNTS' in full_df.columns:
        full_df['log_visits'] = np.log1p(full_df['RAW_VISIT_COUNTS'])

    if 'DISTANCE_FROM_HOME' in full_df.columns:
        full_df['DISTANCE_FROM_HOME'] = pd.to_numeric(full_df['DISTANCE_FROM_HOME'], errors='coerce')

    # --- 3. Save Cache for Next Time ---
    print(f"Saving data to {OUTPUT_FILE} for faster access next time...")
    try:
        full_df.to_parquet(OUTPUT_FILE, index=False)
        print("Save Success!")
    except ImportError:
        print("Note: 'pyarrow' or 'fastparquet' not installed. Saving as pickle instead.")
        full_df.to_pickle(OUTPUT_FILE.replace('.parquet', '.pkl'))
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    print("Data Loading Complete.")
    return full_df