#!/usr/bin/env python3
"""
Contact Force Curve Analysis Script
Specialized for analyzing and visualizing contact force data from CSV files
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
        
        # Basic data check
        print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
        print(f"Number of contact points: {df['contact_id'].nunique()}")
        print(f"Number of geometries: {df['geom1_name'].nunique()}")
        
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def analyze_contact_forces(df):
    """Analyze contact force data"""
    print("\n=== Contact Force Statistical Analysis ===")
    
    # Calculate force magnitude
    df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
    
    # Statistics for each component
    force_cols = ['force_x', 'force_y', 'force_z', 'force_magnitude']
    for col in force_cols:
        if col in df.columns:
            print(f"{col}:")
            print(f"  Mean: {df[col].mean():.3f} N")
            print(f"  Max: {df[col].max():.3f} N")
            print(f"  Min: {df[col].min():.3f} N")
            print(f"  Std: {df[col].std():.3f} N")
    
    return df

def plot_contact_force_curves(df, save_path=None):
    """Plot contact force curves"""
    print("\nGenerating contact force curve plots...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Contact Force Curve Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series - components
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['force_x'], label='Fx', alpha=0.8, linewidth=1)
    ax1.plot(df['timestamp'], df['force_y'], label='Fy', alpha=0.8, linewidth=1)
    ax1.plot(df['timestamp'], df['force_z'], label='Fz', alpha=0.8, linewidth=1)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Contact Force (N)')
    ax1.set_title('Contact Force Components Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series - force magnitude
    ax2 = axes[0, 1]
    ax2.plot(df['timestamp'], df['force_magnitude'], color='red', alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Force Magnitude (N)')
    ax2.set_title('Contact Force Magnitude Time Series')
    ax2.grid(True, alpha=0.3)
    
    # 3. Force distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(df['force_magnitude'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.set_xlabel('Force Magnitude (N)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Contact Force Magnitude Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Component distribution comparison
    ax4 = axes[1, 1]
    force_data = [df['force_x'], df['force_y'], df['force_z']]
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['blue', 'green', 'red']
    
    ax4.boxplot(force_data, labels=labels, patch_artist=True)
    for patch, color in zip(ax4.artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Contact Force (N)')
    ax4.set_title('Force Component Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_contact_analysis(df, save_path=None):
    """Plot contact point analysis"""
    print("\nGenerating contact point analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Contact Point Analysis', fontsize=16, fontweight='bold')
    
    # 1. Contact point count time series
    ax1 = axes[0, 0]
    contact_counts = df.groupby('timestamp')['contact_id'].count()
    ax1.plot(contact_counts.index, contact_counts.values, color='purple', alpha=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Number of Contact Points')
    ax1.set_title('Contact Points per Frame')
    ax1.grid(True, alpha=0.3)
    
    # 2. Geometry contact frequency
    ax2 = axes[0, 1]
    geom_counts = df['geom1_name'].value_counts().head(10)
    bars = ax2.bar(range(len(geom_counts)), geom_counts.values, color='orange', alpha=0.7)
    ax2.set_xlabel('Geometry')
    ax2.set_ylabel('Contact Count')
    ax2.set_title('Geometry Contact Frequency (Top 10)')
    ax2.set_xticks(range(len(geom_counts)))
    ax2.set_xticklabels(geom_counts.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Force vs time scatter plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['timestamp'], df['force_magnitude'], 
                         c=df['force_magnitude'], cmap='viridis', alpha=0.6, s=10)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Force Magnitude (N)')
    ax3.set_title('Contact Force vs Time Scatter Plot')
    plt.colorbar(scatter, ax=ax3, label='Force Magnitude (N)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Force distribution heatmap
    ax4 = axes[1, 1]
    # Create 2D histogram of time vs force
    time_bins = np.linspace(df['timestamp'].min(), df['timestamp'].max(), 50)
    force_bins = np.linspace(0, df['force_magnitude'].max(), 30)
    
    hist, xedges, yedges = np.histogram2d(df['timestamp'], df['force_magnitude'], 
                                         bins=[time_bins, force_bins])
    
    im = ax4.imshow(hist.T, origin='lower', aspect='auto', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    cmap='hot')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Force Magnitude (N)')
    ax4.set_title('Contact Force-Time Distribution Heatmap')
    plt.colorbar(im, ax=ax4, label='Frequency')
    
    plt.tight_layout()
    
    if save_path:
        save_path_analysis = save_path.replace('.png', '_analysis.png')
        plt.savefig(save_path_analysis, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {save_path_analysis}")
    
    plt.show()

def generate_summary_report(df, csv_file):
    """Generate analysis report"""
    print("\n" + "="*60)
    print("Contact Force Data Analysis Report")
    print("="*60)
    
    # Basic statistics
    print(f"Data file: {os.path.basename(csv_file)}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of rows: {len(df):,}")
    print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
    print(f"Total duration: {df['timestamp'].max() - df['timestamp'].min():.3f} seconds")
    
    # Contact force statistics
    df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
    print(f"\nContact Force Statistics:")
    print(f"Average force magnitude: {df['force_magnitude'].mean():.3f} N")
    print(f"Maximum force magnitude: {df['force_magnitude'].max():.3f} N")
    print(f"Minimum force magnitude: {df['force_magnitude'].min():.3f} N")
    print(f"Force magnitude std dev: {df['force_magnitude'].std():.3f} N")
    
    # Contact point statistics
    print(f"\nContact Point Statistics:")
    print(f"Total contact points: {df['contact_id'].nunique()}")
    print(f"Number of geometries: {df['geom1_name'].nunique()}")
    print(f"Average contacts per frame: {df.groupby('timestamp')['contact_id'].count().mean():.2f}")
    
    # Main geometries
    print(f"\nMain Contact Geometries:")
    geom_counts = df['geom1_name'].value_counts().head(5)
    for geom, count in geom_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {geom}: {count} times ({percentage:.1f}%)")
    
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
    
    # Analyze contact forces
    df = analyze_contact_forces(df)
    
    # Generate plots
    plot_file = csv_file.replace('.csv', '_force_curves.png')
    plot_contact_force_curves(df, plot_file)
    
    plot_contact_analysis(df, plot_file)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 