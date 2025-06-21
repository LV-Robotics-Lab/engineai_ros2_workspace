#!/usr/bin/env python3
"""
Contact Point Data Analysis Script
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

def analyze_contact_data(csv_file):
    """Analyze contact point data"""
    print(f"Analyzing file: {csv_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
    print(f"Total duration: {df['timestamp'].max() - df['timestamp'].min():.3f} seconds")
    print(f"Number of contact points: {df['contact_id'].nunique()}")
    print(f"Number of geometry pairs: {df.groupby(['geom1_name', 'geom2_name']).size().count()}")
    
    # Contact force statistics
    print("\n=== Contact Force Statistics (N) ===")
    force_cols = ['force_x', 'force_y', 'force_z']
    for col in force_cols:
        if col in df.columns:
            print(f"{col}: Mean={df[col].mean():.3f}, Max={df[col].max():.3f}, Min={df[col].min():.3f}")
    
    # Contact point position statistics
    print("\n=== Contact Point Position Statistics (m) ===")
    pos_cols = ['pos_x', 'pos_y', 'pos_z']
    for col in pos_cols:
        if col in df.columns:
            print(f"{col}: Mean={df[col].mean():.3f}, Max={df[col].max():.3f}, Min={df[col].min():.3f}")
    
    # Geometry contact frequency
    print("\n=== Geometry Contact Frequency ===")
    geom_counts = df['geom1_name'].value_counts()
    for geom, count in geom_counts.head(10).items():
        print(f"{geom}: {count} contacts")
    
    # Create plots
    create_plots(df, csv_file)

def create_plots(df, csv_file):
    """Create analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Contact Point Data Analysis - {os.path.basename(csv_file)}', fontsize=16)
    
    # 1. Contact force time series
    if 'force_z' in df.columns:
        axes[0, 0].plot(df['timestamp'], df['force_z'], alpha=0.7)
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Contact Force Z (N)')
        axes[0, 0].set_title('Contact Force Z Component Time Series')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Contact point count time series
    contact_counts = df.groupby('timestamp')['contact_id'].count()
    axes[0, 1].plot(contact_counts.index, contact_counts.values, alpha=0.7)
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Number of Contact Points')
    axes[0, 1].set_title('Contact Points per Frame')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Contact force distribution histogram
    if 'force_z' in df.columns:
        axes[1, 0].hist(df['force_z'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Contact Force Z (N)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Contact Force Z Component Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Geometry contact frequency
    geom_counts = df['geom1_name'].value_counts().head(10)
    axes[1, 1].bar(range(len(geom_counts)), geom_counts.values)
    axes[1, 1].set_xlabel('Geometry')
    axes[1, 1].set_ylabel('Contact Count')
    axes[1, 1].set_title('Geometry Contact Frequency (Top 10)')
    axes[1, 1].set_xticks(range(len(geom_counts)))
    axes[1, 1].set_xticklabels(geom_counts.index, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {plot_file}")
    
    # Show plot
    plt.show()

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_contact_data.py <csv_file>")
        print("Example: python3 analyze_contact_data.py logs/contact_data_20231201_143022.csv")
        
        # Auto-find latest CSV file
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            csv_files = [f for f in os.listdir(logs_dir) if f.endswith('.csv') and f.startswith('contact_data_')]
            if csv_files:
                latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(logs_dir, x)))
                csv_file = os.path.join(logs_dir, latest_file)
                print(f"\nFound latest CSV file: {csv_file}")
                print("Analyze this file? (y/n): ", end="")
                if input().lower() == 'y':
                    analyze_contact_data(csv_file)
                return
            else:
                print("No CSV files found")
        return
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    analyze_contact_data(csv_file)

if __name__ == "__main__":
    main() 