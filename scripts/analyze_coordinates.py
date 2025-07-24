#!/usr/bin/env python3
"""
Analyze coordinate differences in contact force CSV data
"""

import pandas as pd
import numpy as np

def analyze_coordinates(csv_file):
    """Analyze coordinate differences between world and URDF coordinates"""
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Check if both coordinate systems are available
    if 'pos_x' in df.columns and 'urdf_x_body1' in df.columns:
        print("\n=== Coordinate Analysis ===")
        
        # Sample some data for analysis
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        print(f"\nAnalyzing {sample_size} sample rows...")
        
        # Compare world vs URDF coordinates
        pos_diff_x = df_sample['pos_x'] - df_sample['urdf_x_body1']
        pos_diff_y = df_sample['pos_y'] - df_sample['urdf_y_body1']
        pos_diff_z = df_sample['pos_z'] - df_sample['urdf_z_body1']
        
        print(f"\nWorld vs URDF coordinate differences:")
        print(f"  X difference: mean={pos_diff_x.mean():.6f}, std={pos_diff_x.std():.6f}")
        print(f"  Y difference: mean={pos_diff_y.mean():.6f}, std={pos_diff_y.std():.6f}")
        print(f"  Z difference: mean={pos_diff_z.mean():.6f}, std={pos_diff_z.std():.6f}")
        
        # Check if coordinates are identical
        identical_x = (pos_diff_x == 0).sum()
        identical_y = (pos_diff_y == 0).sum()
        identical_z = (pos_diff_z == 0).sum()
        
        print(f"\nIdentical coordinates:")
        print(f"  X: {identical_x}/{sample_size} ({identical_x/sample_size*100:.1f}%)")
        print(f"  Y: {identical_y}/{sample_size} ({identical_y/sample_size*100:.1f}%)")
        print(f"  Z: {identical_z}/{sample_size} ({identical_z/sample_size*100:.1f}%)")
        
        # Analyze coordinate ranges
        print(f"\nCoordinate ranges:")
        print(f"  World X: {df_sample['pos_x'].min():.6f} to {df_sample['pos_x'].max():.6f}")
        print(f"  World Y: {df_sample['pos_y'].min():.6f} to {df_sample['pos_y'].max():.6f}")
        print(f"  World Z: {df_sample['pos_z'].min():.6f} to {df_sample['pos_z'].max():.6f}")
        print(f"  URDF X: {df_sample['urdf_x_body1'].min():.6f} to {df_sample['urdf_x_body1'].max():.6f}")
        print(f"  URDF Y: {df_sample['urdf_y_body1'].min():.6f} to {df_sample['urdf_y_body1'].max():.6f}")
        print(f"  URDF Z: {df_sample['urdf_z_body1'].min():.6f} to {df_sample['urdf_z_body1'].max():.6f}")
        
        # Show some examples
        print(f"\nSample coordinate pairs:")
        for i in range(min(5, sample_size)):
            print(f"  Row {i}:")
            print(f"    World: ({df_sample.iloc[i]['pos_x']:.6f}, {df_sample.iloc[i]['pos_y']:.6f}, {df_sample.iloc[i]['pos_z']:.6f})")
            print(f"    URDF:  ({df_sample.iloc[i]['urdf_x_body1']:.6f}, {df_sample.iloc[i]['urdf_y_body1']:.6f}, {df_sample.iloc[i]['urdf_z_body1']:.6f})")
            print(f"    Diff:  ({pos_diff_x.iloc[i]:.6f}, {pos_diff_y.iloc[i]:.6f}, {pos_diff_z.iloc[i]:.6f})")
            print()
        
        # Check for non-zero differences
        non_zero_diff = (pos_diff_x != 0) | (pos_diff_y != 0) | (pos_diff_z != 0)
        if non_zero_diff.any():
            print(f"Found {non_zero_diff.sum()} rows with different coordinates")
            diff_rows = df_sample[non_zero_diff]
            print(f"First few rows with differences:")
            for i in range(min(3, len(diff_rows))):
                row = diff_rows.iloc[i]
                print(f"  Row: World=({row['pos_x']:.6f}, {row['pos_y']:.6f}, {row['pos_z']:.6f})")
                print(f"       URDF=({row['urdf_x_body1']:.6f}, {row['urdf_y_body1']:.6f}, {row['urdf_z_body1']:.6f})")
        else:
            print("All coordinates are identical!")
            
    else:
        print("Required coordinate columns not found in CSV")
        print("Available columns:", list(df.columns))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_coordinates.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_coordinates(csv_file) 