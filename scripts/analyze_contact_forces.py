#!/usr/bin/env python3
"""
Contact Force Analysis Script
Specialized for analyzing contact force data from CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime
import seaborn as sns

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data(csv_file):
    """Load and clean CSV data"""
    print(f"Loading data: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} rows")
        
        # Check if URDF coordinates are available
        urdf_columns = [col for col in df.columns if 'urdf' in col.lower()]
        if urdf_columns:
            print(f"Found URDF coordinate columns: {urdf_columns}")
        else:
            print("Warning: No URDF coordinate columns found")
        
        # Basic data check
        print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
        print(f"Number of contact points: {df['contact_id'].nunique()}")
        print(f"Number of geometries: {df['geom1_name'].nunique()}")
        
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def analyze_total_force(df):
    """Analyze total force magnitude only"""
    print("\n=== Total Force Magnitude Analysis ===")
    
    # Calculate force magnitude if not already present
    if 'force_magnitude' not in df.columns:
        df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
    
    # Statistics for total force magnitude
    print(f"Total Force Magnitude Statistics:")
    print(f"  Mean: {df['force_magnitude'].mean():.3f} N")
    print(f"  Max: {df['force_magnitude'].max():.3f} N")
    print(f"  Min: {df['force_magnitude'].min():.3f} N")
    print(f"  Std: {df['force_magnitude'].std():.3f} N")
    print(f"  Median: {df['force_magnitude'].median():.3f} N")
    
    # Time-based statistics
    print(f"\nTime-based Force Statistics:")
    time_stats = df.groupby('timestamp')['force_magnitude'].agg(['mean', 'max', 'sum'])
    print(f"  Average force per frame: {time_stats['mean'].mean():.3f} N")
    print(f"  Maximum force per frame: {time_stats['max'].max():.3f} N")
    print(f"  Total force over time: {time_stats['sum'].sum():.3f} N")
    
    return df

def plot_total_force_analysis(df, save_path=None):
    """Plot total force analysis"""
    print("\nGenerating total force analysis plots...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Total Force Magnitude Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series - total force magnitude
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['force_magnitude'], color='red', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Force Magnitude (N)')
    ax1.set_title('Total Force Magnitude Time Series')
    ax1.grid(True, alpha=0.3)
    
    # 2. Force distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(df['force_magnitude'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.set_xlabel('Total Force Magnitude (N)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Total Force Magnitude Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Force vs time scatter plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['timestamp'], df['force_magnitude'], 
                         c=df['force_magnitude'], cmap='viridis', alpha=0.6, s=10)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Total Force Magnitude (N)')
    ax3.set_title('Total Force vs Time Scatter Plot')
    plt.colorbar(scatter, ax=ax3, label='Force Magnitude (N)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Force distribution heatmap (time vs force)
    ax4 = axes[1, 1]
    time_bins = np.linspace(df['timestamp'].min(), df['timestamp'].max(), 50)
    force_bins = np.linspace(0, df['force_magnitude'].max(), 30)
    
    hist, xedges, yedges = np.histogram2d(df['timestamp'], df['force_magnitude'], 
                                         bins=[time_bins, force_bins])
    
    im = ax4.imshow(hist.T, origin='lower', aspect='auto', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    cmap='hot')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Total Force Magnitude (N)')
    ax4.set_title('Total Force-Time Distribution Heatmap')
    plt.colorbar(im, ax=ax4, label='Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Total force plot saved to: {save_path}")
    
    plt.show()

def create_time_evolution_heatmap(df, save_path=None):
    """Create time evolution heatmap showing force changes over time and space"""
    print("\nGenerating time evolution heatmap...")
    
    # Check if URDF coordinates are available
    urdf_columns = [col for col in df.columns if 'urdf' in col.lower()]
    if not urdf_columns:
        print("Warning: No URDF coordinate columns found. Skipping time evolution heatmap.")
        return
    
    # Use body1 coordinates
    if 'urdf_x_body1' in df.columns and 'urdf_y_body1' in df.columns:
        x_col, y_col = 'urdf_x_body1', 'urdf_y_body1'
    elif 'urdf_x_body2' in df.columns and 'urdf_y_body2' in df.columns:
        x_col, y_col = 'urdf_x_body2', 'urdf_y_body2'
    else:
        print("Warning: No 2D URDF coordinates found. Skipping time evolution heatmap.")
        return
    
    # Filter valid data
    valid_mask = (df[x_col].notna() & df[y_col].notna() & 
                  np.isfinite(df[x_col]) & np.isfinite(df[y_col]))
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("Warning: No valid data for time evolution heatmap.")
        return
    
    # Create 2D histogram
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Contact Force Time Evolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spatial force distribution (2D heatmap)
    ax1 = axes[0, 0]
    x_bins = np.linspace(df_valid[x_col].min(), df_valid[x_col].max(), 30)
    y_bins = np.linspace(df_valid[y_col].min(), df_valid[y_col].max(), 30)
    
    hist, xedges, yedges = np.histogram2d(df_valid[x_col], df_valid[y_col], 
                                         bins=[x_bins, y_bins], 
                                         weights=df_valid['force_magnitude'])
    counts, _, _ = np.histogram2d(df_valid[x_col], df_valid[y_col], 
                                 bins=[x_bins, y_bins])
    
    # Average force per bin
    avg_force = np.divide(hist, counts, out=np.zeros_like(hist), where=counts>0)
    
    im1 = ax1.imshow(avg_force.T, origin='lower', aspect='auto',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     cmap='hot')
    ax1.set_xlabel(f'URDF X')
    ax1.set_ylabel(f'URDF Y')
    ax1.set_title('Average Force Distribution (2D)')
    plt.colorbar(im1, ax=ax1, label='Average Force (N)')
    
    # 2. Time vs position heatmap
    ax2 = axes[0, 1]
    time_bins = np.linspace(df_valid['timestamp'].min(), df_valid['timestamp'].max(), 50)
    pos_bins = np.linspace(df_valid[x_col].min(), df_valid[x_col].max(), 30)
    
    hist2, xedges2, yedges2 = np.histogram2d(df_valid['timestamp'], df_valid[x_col], 
                                             bins=[time_bins, pos_bins],
                                             weights=df_valid['force_magnitude'])
    counts2, _, _ = np.histogram2d(df_valid['timestamp'], df_valid[x_col], 
                                  bins=[time_bins, pos_bins])
    
    avg_force2 = np.divide(hist2, counts2, out=np.zeros_like(hist2), where=counts2>0)
    
    im2 = ax2.imshow(avg_force2.T, origin='lower', aspect='auto',
                     extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], 
                     cmap='viridis')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel(f'URDF X Position')
    ax2.set_title('Force Evolution Over Time')
    plt.colorbar(im2, ax=ax2, label='Average Force (N)')
    
    # 3. Force magnitude distribution over time
    ax3 = axes[1, 0]
    time_groups = df_valid.groupby(pd.cut(df_valid['timestamp'], bins=20))
    time_means = time_groups['force_magnitude'].mean()
    time_stds = time_groups['force_magnitude'].std()
    
    ax3.errorbar(time_means.index.astype(str), time_means.values, 
                yerr=time_stds.values, fmt='o-', capsize=5)
    ax3.set_xlabel('Time Bins')
    ax3.set_ylabel('Average Force Magnitude (N)')
    ax3.set_title('Force Magnitude Evolution Over Time')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Contact point density
    ax4 = axes[1, 1]
    hist_density, _, _ = np.histogram2d(df_valid[x_col], df_valid[y_col], 
                                       bins=[x_bins, y_bins])
    
    im4 = ax4.imshow(hist_density.T, origin='lower', aspect='auto',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     cmap='Blues')
    ax4.set_xlabel(f'URDF X')
    ax4.set_ylabel(f'URDF Y')
    ax4.set_title('Contact Point Density')
    plt.colorbar(im4, ax=ax4, label='Contact Count')
    
    plt.tight_layout()
    
    if save_path:
        evolution_path = save_path.replace('.png', '_time_evolution.png')
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        print(f"Time evolution heatmap saved to: {evolution_path}")
    
    plt.show()

def generate_summary_report(df, csv_file):
    """Generate analysis report focusing on total force"""
    print("\n" + "="*60)
    print("Contact Force Analysis Report (Total Force Focus)")
    print("="*60)
    
    # Basic statistics
    print(f"Data file: {os.path.basename(csv_file)}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of rows: {len(df):,}")
    print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
    print(f"Total duration: {df['timestamp'].max() - df['timestamp'].min():.3f} seconds")
    
    # Total force statistics
    if 'force_magnitude' not in df.columns:
        df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
    
    print(f"\nTotal Force Statistics:")
    print(f"Average force magnitude: {df['force_magnitude'].mean():.3f} N")
    print(f"Maximum force magnitude: {df['force_magnitude'].max():.3f} N")
    print(f"Minimum force magnitude: {df['force_magnitude'].min():.3f} N")
    print(f"Force magnitude std dev: {df['force_magnitude'].std():.3f} N")
    print(f"Median force magnitude: {df['force_magnitude'].median():.3f} N")
    
    # Time-based statistics
    time_stats = df.groupby('timestamp')['force_magnitude'].agg(['mean', 'max', 'sum'])
    print(f"\nTime-based Statistics:")
    print(f"Average force per frame: {time_stats['mean'].mean():.3f} N")
    print(f"Maximum force per frame: {time_stats['max'].max():.3f} N")
    print(f"Total force over time: {time_stats['sum'].sum():.3f} N")
    
    # URDF coordinate statistics
    urdf_columns = [col for col in df.columns if 'urdf' in col.lower()]
    if urdf_columns:
        print(f"\nURDF Coordinate Analysis:")
        for col in urdf_columns:
            if col in df.columns and df[col].notna().any():
                valid_data = df[col].dropna()
                print(f"  {col}: range [{valid_data.min():.3f}, {valid_data.max():.3f}]")
    
    print("="*60)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_contact_forces.py <csv_file>")
        print("Example: python3 analyze_contact_forces.py logs/contact_data_20231201_143022.csv")
        
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
                    analyze_csv_file(csv_file)
                return
            else:
                print("No CSV files found")
        return
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    analyze_csv_file(csv_file)

def analyze_csv_file(csv_file):
    """Analyze CSV file"""
    # Load data
    df = load_and_clean_data(csv_file)
    if df is None:
        return
    
    # Generate report
    generate_summary_report(df, csv_file)
    
    # Analyze total force
    df = analyze_total_force(df)
    
    # Generate plots
    plot_file = csv_file.replace('.csv', '_total_force_analysis.png')
    plot_total_force_analysis(df, plot_file)
    
    # Create time evolution heatmap
    create_time_evolution_heatmap(df, plot_file)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 