"""
Script to fix data type issues in existing CSV files
Run this if you encounter format errors with existing data
"""
import pandas as pd
import os
import glob
from paths import DATA_DIR

def fix_csv_types(file_path):
    """Fix numeric column types in CSV files"""
    try:
        df = pd.read_csv(file_path)
        
        # Identify numeric columns
        numeric_cols = []
        for col in df.columns:
            if col.lower() in ['close', 'open', 'high', 'low', 'volume', 'adj close', 
                              'rolling_sentiment', 'sentiment_score', 'confidence']:
                numeric_cols.append(col)
        
        # Convert to numeric
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save back
        df.to_csv(file_path, index=False)
        print(f"✅ Fixed: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all CSV files in data directory"""
    data_dir = DATA_DIR
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return
    
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to check...")
    fixed = 0    
    for csv_file in csv_files:
        if fix_csv_types(csv_file):
            fixed += 1
    
    print(f"\n✨ Fixed {fixed}/{len(csv_files)} files")

if __name__ == "__main__":
    main()
