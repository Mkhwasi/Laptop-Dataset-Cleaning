"""
Laptop Dataset Cleaning & Feature Engineering Pipeline
------------------------------------------------------
Usage:
    python clean_data.py <input_raw_csv> <output_clean_csv>

Example:
    python clean_data.py laptop_data.csv clean_laptop_data.csv
"""

import sys
import re
import pandas as pd
import numpy as np


def load_and_audit_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV and standardizes missing value representations."""
    df = pd.read_csv(filepath)
    # Replace common placeholder strings with formal NaN
    df.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df


def clean_basic_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips unit strings and handles numeric types for Inches, Ram, Weight, and Price."""
    # 1. Inches
    if 'Inches' in df.columns:
        df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
        # Impute single missing values with median of the laptop TypeName group
        if df['Inches'].isnull().sum() > 0:
            df['Inches'] = df.groupby('TypeName')['Inches'].transform(
                lambda x: x.fillna(x.median())
            )

    # 2. Ram (e.g., '8GB' -> 8)
    if 'Ram' in df.columns:
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', case=False).str.strip()
        df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')

    # 3. Weight (e.g., '1.37kg' -> 1.37)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', case=False).str.strip()
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

    # 4. Price
    if 'Price' in df.columns and df['Price'].dtype == 'object':
        df['Price'] = df['Price'].astype(str).str.replace(r'[$€,]', '', regex=True).str.strip()
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    return df


def engineer_screen_resolution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts Touchscreen, IPS panel flag, and Pixels Per Inch (PPI)."""
    if 'ScreenResolution' not in df.columns or 'Inches' not in df.columns:
        return df

    res_str = df['ScreenResolution'].astype(str)

    # Flags
    df['Touchscreen'] = res_str.apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS'] = res_str.apply(lambda x: 1 if 'IPS' in x else 0)

    # Extract X and Y pixel dimensions
    resolution = res_str.str.extract(r'(\d+)x(\d+)')
    x_res = resolution[0].astype(float)
    y_res = resolution[1].astype(float)

    # Calculate Pixels Per Inch (PPI)
    df['PPI'] = (np.sqrt(x_res**2 + y_res**2) / df['Inches']).round(2)

    return df


def engineer_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts CPU Clock Speed and CPU Brand/Tier."""
    if 'Cpu' not in df.columns:
        return df

    cpu_str = df['Cpu'].astype(str)

    # Extract clock speed (GHz)
    df['Clock_Speed_GHz'] = cpu_str.apply(
        lambda x: float(re.search(r'([\d\.]+)GHz', x).group(1)) if re.search(r'([\d\.]+)GHz', x) else np.nan
    )

    # Categorize CPU Brand/Tier
    def fetch_processor(text):
        if 'Intel Core i7' in text:
            return 'Intel Core i7'
        elif 'Intel Core i5' in text:
            return 'Intel Core i5'
        elif 'Intel Core i3' in text:
            return 'Intel Core i3'
        elif 'Intel' in text:
            return 'Other Intel'
        elif 'AMD' in text:
            return 'AMD'
        return 'Other'

    df['Cpu_Brand'] = cpu_str.apply(fetch_processor)

    return df


def engineer_gpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw GPU strings and extracts Brand, Type, and Tier."""
    if 'Gpu' not in df.columns:
        return df

    def clean_gpu_string(gpu_str):
        if pd.isna(gpu_str):
            return ""
        gpu_str = str(gpu_str).strip()
        # Fix encoding corruption (e.g. <U+039C> -> M)
        gpu_str = re.sub(r'<U\+[0-9A-Fa-f]+>', 'M', gpu_str)
        # Fix missing spaces in model numbers (GTX1080 -> GTX 1080)
        gpu_str = re.sub(r'GTX\s*(\d+)', r'GTX \1', gpu_str, flags=re.IGNORECASE)
        return gpu_str

    cleaned_gpu = df['Gpu'].apply(clean_gpu_string)

    # 1. Brand
    def get_brand(x):
        if 'Nvidia' in x:
            return 'Nvidia'
        elif 'Intel' in x:
            return 'Intel'
        elif 'AMD' in x:
            return 'AMD'
        return 'Other'

    df['Gpu_Brand'] = cleaned_gpu.apply(get_brand)

    # 2. Type (Integrated vs Dedicated)
    def get_type(x):
        if 'Intel' in x:
            return 'Integrated'
        elif any(term in x for term in ['GeForce', 'Quadro', 'GTX', 'MX', 'Radeon', 'FirePro']):
            return 'Dedicated'
        return 'Integrated'

    df['Gpu_Type'] = cleaned_gpu.apply(get_type)

    # 3. Performance Tier
    def get_tier(x):
        if 'Intel' in x:
            if 'Iris' in x:
                return 'Intel Iris (High Integrated)'
            return 'Intel HD/UHD (Standard Integrated)'
        elif 'Nvidia' in x:
            if 'Quadro' in x:
                return 'Nvidia Quadro (Workstation)'
            elif any(gtx in x for gtx in ['GTX 1080', 'GTX 1070', 'GTX 1060', 'GTX 980', 'GTX 970', 'SLI']):
                return 'Nvidia High Gaming'
            elif any(gtx in x for gtx in ['GTX 1050', 'GTX 965', 'GTX 960', 'GTX 950']):
                return 'Nvidia Mid Gaming'
            else:
                return 'Nvidia Entry/Casual'
        elif 'AMD' in x:
            if 'FirePro' in x or 'Radeon Pro' in x:
                return 'AMD Pro/Workstation'
            elif any(rx in x for rx in ['RX 580', 'RX 570', 'RX 560', 'R9']):
                return 'AMD Gaming'
            else:
                return 'AMD Entry/Casual'
        return 'Other'

    df['Gpu_Tier'] = cleaned_gpu.apply(get_tier)

    return df


def engineer_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts storage capacities (SSD, HDD, Flash, Hybrid) in GB."""
    if 'Memory' not in df.columns:
        return df

    def parse_memory(mem_str):
        mem_str = str(mem_str).replace('.0', '')
        ssd, hdd, flash, hybrid = 0, 0, 0, 0

        # Split multiple storage drives (e.g. 128GB SSD + 1TB HDD)
        parts = mem_str.split('+')
        for part in parts:
            part = part.strip()
            match = re.search(r'(\d+)\s*(GB|TB)\s*(SSD|HDD|Flash Storage|Hybrid)', part, re.IGNORECASE)
            if match:
                size = int(match.group(1))
                unit = match.group(2).upper()
                drive_type = match.group(3).title()

                # Convert TB to GB
                if unit == 'TB':
                    size *= 1024

                if 'Ssd' in drive_type:
                    ssd += size
                elif 'Hdd' in drive_type:
                    hdd += size
                elif 'Flash' in drive_type:
                    flash += size
                elif 'Hybrid' in drive_type:
                    hybrid += size

        return pd.Series([ssd, hdd, flash, hybrid])

    df[['SSD_GB', 'HDD_GB', 'Flash_GB', 'Hybrid_GB']] = df['Memory'].apply(parse_memory)
    return df


def consolidate_opsys(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidates fragmented operating system categories."""
    if 'OpSys' not in df.columns:
        return df

    def clean_os(os_str):
        if pd.isna(os_str):
            return 'Other / No OS'
        os_str = str(os_str).strip()
        if os_str in ['macOS', 'Mac OS X']:
            return 'Mac'
        elif 'Windows' in os_str:
            return 'Windows'
        elif os_str == 'Linux':
            return 'Linux'
        elif os_str == 'Chrome OS':
            return 'Chrome OS'
        return 'Other / No OS'

    df['OpSys'] = df['OpSys'].apply(clean_os)
    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespaces, drops raw/redundant columns, and resets index."""
    # Strip whitespace from nominal text columns
    text_cols = ['Company', 'TypeName']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop raw unparsed text columns
    raw_cols_to_drop = ['ScreenResolution', 'Cpu', 'Gpu', 'Memory']
    df.drop(columns=[col for col in raw_cols_to_drop if col in df.columns], inplace=True)

    # Remove duplicates and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def run_pipeline(input_path: str, output_path: str):
    """Executes full cleaning and feature engineering workflow."""
    print(f"Loading dataset from: {input_path}")
    df = load_and_audit_data(input_path)

    print("Cleaning numeric fields...")
    df = clean_basic_numeric_columns(df)

    print("Engineering Screen Resolution features (IPS, Touchscreen, PPI)...")
    df = engineer_screen_resolution_features(df)

    print("Engineering CPU features...")
    df = engineer_cpu_features(df)

    print("Engineering GPU features...")
    df = engineer_gpu_features(df)

    print("Engineering Storage/Memory features...")
    df = engineer_memory_features(df)

    print("Consolidating Operating Systems...")
    df = consolidate_opsys(df)

    print("Finalizing pipeline and dropping raw text columns...")
    df = finalize_dataset(df)

    print(f"Exporting clean dataset ({df.shape[0]} rows × {df.shape[1]} columns) to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Data cleaning pipeline completed successfully!")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Error: Missing file paths.")
        print("Usage: python clean_data.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    run_pipeline(input_csv, output_csv)