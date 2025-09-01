#!/usr/bin/env python3
"""
Contact Force Analysis Script
Specialized for analyzing contact force data from CSV files
Updated for new CSV format with collision link pose and force_normal
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
        # 尝试不同的CSV读取方式来处理换行问题
        try:
            # 首先尝试标准读取
            df = pd.read_csv(csv_file)
        except:
            # 如果失败，尝试处理换行问题
            print("Standard CSV reading failed, trying to handle line breaks...")
            with open(csv_file, 'r') as f:
                content = f.read()
            
            # 修复换行的列名
            lines = content.split('\n')
            header = []
            current_header = ""
            
            for line in lines:
                if line.strip():
                    if current_header:
                        current_header += line.strip()
                    else:
                        current_header = line.strip()
                    
                    if current_header.count(',') >= 55:  # 应该有56列
                        header.append(current_header)
                        current_header = ""
                        break
            
            if header:
                # 重新读取，使用修复后的头部
                df = pd.read_csv(csv_file, header=None, names=header[0].split(','))
                print("Successfully loaded with fixed header")
            else:
                # 如果还是失败，使用默认列名
                df = pd.read_csv(csv_file, header=None)
                print("Loaded with default column names")
        
        print(f"Successfully loaded {len(df)} rows")
        
        # Check if robot frame coordinates are available
        robot_frame_columns = [col for col in df.columns if 'robot_frame' in col.lower()]
        if robot_frame_columns:
            print(f"Found robot frame coordinate columns: {robot_frame_columns}")
        else:
            print("Warning: No robot frame coordinate columns found")
        
        # Check if collision link pose columns are available
        collision_link_columns = [col for col in df.columns if 'collision_link' in col.lower()]
        if collision_link_columns:
            print(f"Found collision link pose columns: {collision_link_columns}")
        else:
            print("Warning: No collision link pose columns found")
        
        # Check if force_normal column is available
        if 'force_normal' in df.columns:
            print("Found force_normal column - will analyze normal force component")
        else:
            print("Warning: No force_normal column found")
        
        # Check if joint angles are available
        joint_columns = [col for col in df.columns if 'joint_' in col and '_angle' in col]
        if joint_columns:
            print(f"Found {len(joint_columns)} joint angle columns")
        else:
            print("Warning: No joint angle columns found")
        
        # Basic data check
        print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
        print(f"Number of contact points: {df['contact_id'].nunique()}")
        print(f"Number of body1 names: {df['body1_name'].nunique()}")
        print(f"Number of body2 names: {df['body2_name'].nunique()}")
        
        # Debug: Check force data
        if 'force_magnitude' in df.columns:
            print(f"Force magnitude range: {df['force_magnitude'].min():.3f} - {df['force_magnitude'].max():.3f} N")
            print(f"First few force_magnitude values: {df['force_magnitude'].head().tolist()}")
            print(f"Data type: {df['force_magnitude'].dtype}")
        
        if 'force_normal' in df.columns:
            print(f"Force normal range: {df['force_normal'].min():.3f} - {df['force_normal'].max():.3f} N")
            print(f"First few force_normal values: {df['force_normal'].head().tolist()}")
            print(f"Data type: {df['force_normal'].dtype}")
        
        if 'force_x' in df.columns:
            print(f"Force X range: {df['force_x'].min():.3f} - {df['force_x'].max():.3f} N")
            print(f"First few force_x values: {df['force_x'].head().tolist()}")
        
        # Check for any NaN or infinite values
        if 'force_magnitude' in df.columns:
            nan_count = df['force_magnitude'].isna().sum()
            inf_count = np.isinf(df['force_magnitude']).sum()
            print(f"Force magnitude - NaN: {nan_count}, Inf: {inf_count}")
        
        # Debug: Print all column names to check mapping
        print(f"\nAll column names:")
        for i, col in enumerate(df.columns):
            print(f"  {i:2d}: {col}")
        
        # Debug: Check first few rows of key columns
        print(f"\nFirst row data:")
        if len(df) > 0:
            first_row = df.iloc[0]
            key_cols = ['timestamp', 'contact_id', 'force_magnitude', 'force_normal', 'force_x', 'force_y', 'force_z']
            for col in key_cols:
                if col in df.columns:
                    print(f"  {col}: {first_row[col]}")
        
        # Debug: Check raw CSV reading
        print(f"\nRaw CSV inspection:")
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            print(f"  First line (header): {first_line[:100]}...")
            print(f"  Second line (data): {second_line[:100]}...")
            print(f"  Header comma count: {first_line.count(',')}")
            print(f"  Data comma count: {second_line.count(',')}")
        
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def analyze_force_components(df):
    """Analyze force components including force_normal"""
    print("\n=== Force Components Analysis ===")
    
    # Check if force_normal exists, if not calculate it
    if 'force_normal' not in df.columns:
        print("Warning: force_normal column not found, calculating from contact frame")
        # This would require the original contact frame data which might not be available
        df['force_normal'] = np.nan
    
    # Statistics for force_normal
    if 'force_normal' in df.columns and df['force_normal'].notna().any():
        print(f"Force Normal (f_norm) Statistics:")
        print(f"  Mean: {df['force_normal'].mean():.3f} N")
        print(f"  Max: {df['force_normal'].max():.3f} N")
        print(f"  Min: {df['force_normal'].min():.3f} N")
        print(f"  Std: {df['force_normal'].std():.3f} N")
        print(f"  Median: {df['force_normal'].median():.3f} N")
        
        # Count positive normal forces (actual contact)
        positive_normal = (df['force_normal'] > 0).sum()
        total_contacts = len(df)
        print(f"  Positive normal forces (actual contact): {positive_normal}/{total_contacts} ({100*positive_normal/total_contacts:.1f}%)")
    
    # Statistics for total force magnitude
    if 'force_magnitude' in df.columns:
        print(f"\nTotal Force Magnitude (f_mag) Statistics:")
        print(f"  Mean: {df['force_magnitude'].mean():.3f} N")
        print(f"  Max: {df['force_magnitude'].max():.3f} N")
        print(f"  Min: {df['force_magnitude'].min():.3f} N")
        print(f"  Std: {df['force_magnitude'].std():.3f} N")
        print(f"  Median: {df['force_magnitude'].median():.3f} N")
    else:
        # Calculate force magnitude if not present
        df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
        print(f"\nCalculated Total Force Magnitude Statistics:")
        print(f"  Mean: {df['force_magnitude'].mean():.3f} N")
        print(f"  Max: {df['force_magnitude'].max():.3f} N")
        print(f"  Min: {df['force_magnitude'].min():.3f} N")
    
    # Time-based statistics
    print(f"\nTime-based Force Statistics:")
    time_stats = df.groupby('timestamp')['force_magnitude'].agg(['mean', 'max', 'count'])
    print(f"  Average force per frame: {time_stats['mean'].mean():.3f} N")
    print(f"  Maximum force per frame: {time_stats['max'].max():.3f} N")
    print(f"  Average contact points per frame: {time_stats['count'].mean():.1f}")
    print(f"  Total frames analyzed: {len(time_stats)}")
    
    return df

def plot_force_analysis(df, save_path=None):
    """Plot force analysis including force_normal"""
    print("\nGenerating force analysis plots...")
    
    # Debug: Print data ranges before plotting
    print(f"Plotting data ranges:")
    if 'force_magnitude' in df.columns:
        print(f"  Force magnitude: {df['force_magnitude'].min():.3f} - {df['force_magnitude'].max():.3f} N")
    if 'force_normal' in df.columns:
        print(f"  Force normal: {df['force_normal'].min():.3f} - {df['force_normal'].max():.3f} N")
    if 'force_x' in df.columns:
        print(f"  Force X: {df['force_x'].min():.3f} - {df['force_x'].max():.3f} N")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Contact Force Analysis (Including Force Normal)', fontsize=16, fontweight='bold')
    
    # 1. Time series - total force magnitude
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['force_magnitude'], color='red', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Force Magnitude (N)')
    ax1.set_title('Total Force Magnitude Time Series')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series - force normal
    ax2 = axes[0, 1]
    if 'force_normal' in df.columns and df['force_normal'].notna().any():
        ax2.plot(df['timestamp'], df['force_normal'], color='blue', alpha=0.8, linewidth=1.5)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Force Normal (N)')
        ax2.set_title('Force Normal Time Series')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No force_normal data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Force Normal Time Series (No Data)')
    
    # 3. Force magnitude distribution histogram
    ax3 = axes[0, 2]
    ax3.hist(df['force_magnitude'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.set_xlabel('Total Force Magnitude (N)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Total Force Magnitude Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Force normal distribution histogram
    ax4 = axes[1, 0]
    if 'force_normal' in df.columns and df['force_normal'].notna().any():
        ax4.hist(df['force_normal'], bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
        ax4.set_xlabel('Force Normal (N)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Force Normal Distribution')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No force_normal data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Force Normal Distribution (No Data)')
    
    # 5. Force magnitude vs force normal scatter plot
    ax5 = axes[1, 1]
    if 'force_normal' in df.columns and df['force_normal'].notna().any():
        scatter = ax5.scatter(df['force_normal'], df['force_magnitude'], 
                             c=df['force_magnitude'], cmap='viridis', alpha=0.6, s=10)
        ax5.set_xlabel('Force Normal (N)')
        ax5.set_ylabel('Total Force Magnitude (N)')
        ax5.set_title('Force Magnitude vs Force Normal')
        plt.colorbar(scatter, ax=ax5, label='Force Magnitude (N)')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No force_normal data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Force Magnitude vs Force Normal (No Data)')
    
    # 6. Force components comparison
    ax6 = axes[1, 2]
    force_components = ['force_x', 'force_y', 'force_z']
    if all(col in df.columns for col in force_components):
        component_data = [df[col].abs() for col in force_components]
        ax6.boxplot(component_data, labels=['X', 'Y', 'Z'])
        ax6.set_ylabel('Force Component Magnitude (N)')
        ax6.set_title('Force Components Comparison')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Missing force component data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Force Components Comparison (No Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Force analysis plot saved to: {save_path}")
    
    plt.show()

def create_time_evolution_heatmap(df, save_path=None):
    """Create time evolution heatmap showing force changes over time and space"""
    print("\nGenerating time evolution heatmap...")
    
    # Check if robot frame coordinates are available
    robot_frame_columns = [col for col in df.columns if 'robot_frame' in col.lower()]
    if not robot_frame_columns:
        print("Warning: No robot frame coordinate columns found. Skipping time evolution heatmap.")
        return
    
    # Use robot frame coordinates (new format)
    if 'robot_frame_x' in df.columns and 'robot_frame_y' in df.columns:
        x_col, y_col = 'robot_frame_x', 'robot_frame_y'
    else:
        print("Warning: No 2D robot frame coordinates found. Skipping time evolution heatmap.")
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
    ax1.set_xlabel(f'Robot Frame X')
    ax1.set_ylabel(f'Robot Frame Y')
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
    ax2.set_ylabel(f'Robot Frame X Position')
    ax2.set_title('Force Evolution Over Time')
    plt.colorbar(im2, ax=ax2, label='Average Force (N)')
    
    # 3. Force normal evolution over time
    ax3 = axes[1, 0]
    if 'force_normal' in df_valid.columns and df_valid['force_normal'].notna().any():
        time_groups = df_valid.groupby(pd.cut(df_valid['timestamp'], bins=20), observed=False)
        time_means = time_groups['force_normal'].mean()
        time_stds = time_groups['force_normal'].std()
        
        ax3.errorbar(time_means.index.astype(str), time_means.values, 
                    yerr=time_stds.values, fmt='o-', capsize=5, color='blue')
        ax3.set_xlabel('Time Bins')
        ax3.set_ylabel('Average Force Normal (N)')
        ax3.set_title('Force Normal Evolution Over Time')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No force_normal data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Force Normal Evolution Over Time (No Data)')
    
    # 4. Contact point density
    ax4 = axes[1, 1]
    hist_density, _, _ = np.histogram2d(df_valid[x_col], df_valid[y_col], 
                                       bins=[x_bins, y_bins])
    
    im4 = ax4.imshow(hist_density.T, origin='lower', aspect='auto',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     cmap='Blues')
    ax4.set_xlabel(f'Robot Frame X')
    ax4.set_ylabel(f'Robot Frame Y')
    ax4.set_title('Contact Point Density')
    plt.colorbar(im4, ax=ax4, label='Contact Count')
    
    plt.tight_layout()
    
    if save_path:
        evolution_path = save_path.replace('.png', '_time_evolution.png')
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        print(f"Time evolution heatmap saved to: {evolution_path}")
    
    plt.show()

def analyze_collision_link_pose(df):
    """Analyze collision link pose data"""
    print("\n=== Collision Link Pose Analysis ===")
    
    collision_link_columns = [col for col in df.columns if 'collision_link' in col.lower()]
    if not collision_link_columns:
        print("No collision link pose columns found")
        return
    
    # Analyze collision link positions
    pos_columns = [col for col in collision_link_columns if 'pos' in col]
    if pos_columns:
        print(f"Collision Link Position Analysis:")
        for col in pos_columns:
            if df[col].notna().any():
                valid_data = df[col].dropna()
                print(f"  {col}: range [{valid_data.min():.3f}, {valid_data.max():.3f}], mean: {valid_data.mean():.3f}")
    
    # Analyze collision link orientations
    quat_columns = [col for col in collision_link_columns if 'quat' in col]
    if quat_columns:
        print(f"Collision Link Orientation Analysis:")
        for col in quat_columns:
            if df[col].notna().any():
                valid_data = df[col].dropna()
                print(f"  {col}: range [{valid_data.min():.3f}, {valid_data.max():.3f}], mean: {valid_data.mean():.3f}")

def generate_summary_report(df, csv_file):
    """Generate analysis report focusing on force_normal and new data structure"""
    print("\n" + "="*60)
    print("Contact Force Analysis Report (Updated Format)")
    print("="*60)
    
    # Basic statistics
    print(f"Data file: {os.path.basename(csv_file)}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of rows: {len(df):,}")
    print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
    print(f"Total duration: {df['timestamp'].max() - df['timestamp'].min():.3f} seconds")
    
    # Force statistics
    if 'force_magnitude' not in df.columns:
        df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
    
    print(f"\nForce Statistics:")
    print(f"Average force magnitude: {df['force_magnitude'].mean():.3f} N")
    print(f"Maximum force magnitude: {df['force_magnitude'].max():.3f} N")
    print(f"Minimum force magnitude: {df['force_magnitude'].min():.3f} N")
    print(f"Force magnitude std dev: {df['force_magnitude'].std():.3f} N")
    print(f"Median force magnitude: {df['force_magnitude'].median():.3f} N")
    
    # Force normal statistics
    if 'force_normal' in df.columns and df['force_normal'].notna().any():
        print(f"\nForce Normal Statistics:")
        print(f"Average force normal: {df['force_normal'].mean():.3f} N")
        print(f"Maximum force normal: {df['force_normal'].max():.3f} N")
        print(f"Minimum force normal: {df['force_normal'].min():.3f} N")
        print(f"Force normal std dev: {df['force_normal'].std():.3f} N")
        print(f"Median force normal: {df['force_normal'].median():.3f} N")
        
        # Contact quality analysis
        positive_normal = (df['force_normal'] > 0).sum()
        total_contacts = len(df)
        print(f"Contact quality: {positive_normal}/{total_contacts} ({100*positive_normal/total_contacts:.1f}% positive normal forces)")
    
    # Time-based statistics
    time_stats = df.groupby('timestamp')['force_magnitude'].agg(['mean', 'max', 'count'])
    print(f"\nTime-based Statistics:")
    print(f"Average force per frame: {time_stats['mean'].mean():.3f} N")
    print(f"Maximum force per frame: {time_stats['max'].max():.3f} N")
    print(f"Average contact points per frame: {time_stats['count'].mean():.1f}")
    print(f"Total frames analyzed: {len(time_stats)}")
    
    # Robot frame coordinate statistics
    robot_frame_columns = [col for col in df.columns if 'robot_frame' in col.lower()]
    if robot_frame_columns:
        print(f"\nRobot Frame Coordinate Analysis:")
        for col in robot_frame_columns:
            if col in df.columns and df[col].notna().any():
                valid_data = df[col].dropna()
                print(f"  {col}: range [{valid_data.min():.3f}, {valid_data.max():.3f}]")
    
    # Collision link pose statistics
    collision_link_columns = [col for col in df.columns if 'collision_link' in col.lower()]
    if collision_link_columns:
        print(f"\nCollision Link Pose Analysis:")
        for col in collision_link_columns:
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
    
    # Analyze force components including force_normal
    df = analyze_force_components(df)
    
    # Analyze collision link pose
    analyze_collision_link_pose(df)
    
    # Generate plots
    plot_file = csv_file.replace('.csv', '_force_analysis.png')
    plot_force_analysis(df, plot_file)
    
    # Create time evolution heatmap
    create_time_evolution_heatmap(df, plot_file)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 