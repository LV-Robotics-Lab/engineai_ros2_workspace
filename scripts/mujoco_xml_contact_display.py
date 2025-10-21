#!/usr/bin/env python3
"""
Mujoco XML Contact Force Display
Uses official Mujoco Python interface to display XML model with contact forces as spheres
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tempfile
import xml.etree.ElementTree as ET
import mediapy as media
from datetime import datetime

# Set MuJoCo memory limits to avoid stack overflow - MUST be set before importing mujoco
# os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for better memory management
os.environ['MUJOCO_STACK_SIZE'] = '1073741824'  # 1GB stack size (increased for large models with many spheres)
os.environ['MUJOCO_ARENA_SIZE'] = '8589934592'  # 8GB arena size for constraints (doubled for more spheres)

# Import mujoco AFTER setting environment variables
import mujoco as mj

def analyze_csv_data(df):
    """Analyze CSV data and print statistics"""
    print("\n=== CSV Data Analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for key columns based on new CSV format
    key_columns = {
        'Time': ['timestamp'],
        'Contact Info': ['contact_id', 'body1_name', 'body2_name'],
        'World Position': ['pos_x', 'pos_y', 'pos_z'],
        'Robot Frame Position': ['robot_frame_x', 'robot_frame_y', 'robot_frame_z'],
        'Forces': ['force_x', 'force_y', 'force_z', 'force_magnitude', 'force_normal'],
        'Torques': ['torque_x', 'torque_y', 'torque_z'],
        'Base Link Pose': ['base_link_x', 'base_link_y', 'base_link_z', 'base_link_qw', 'base_link_qx', 'base_link_qy', 'base_link_qz'],
        'Collision Link Pose': ['collision_link_x', 'collision_link_y', 'collision_link_z', 'collision_link_qw', 'collision_link_qx', 'collision_link_qy', 'collision_link_qz'],
        'Joint Angles': [col for col in df.columns if 'joint_' in col and '_angle' in col]
    }
    
    for category, columns in key_columns.items():
        found_columns = [col for col in columns if col in df.columns]
        if found_columns:
            print(f"{category}: {found_columns}")
            if category == 'Time' and 'timestamp' in df.columns:
                print(f"  Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
            elif category == 'Forces' and 'force_magnitude' in df.columns:
                print(f"  Force range: {df['force_magnitude'].min():.3f} - {df['force_magnitude'].max():.3f} N")
                if 'force_normal' in df.columns:
                    print(f"  Normal force range: {df['force_normal'].min():.3f} - {df['force_normal'].max():.3f} N")
            elif category == 'Joint Angles':
                print(f"  Number of joints: {len(found_columns)}")
        else:
            print(f"{category}: Not found")
    
    # Additional analysis for new CSV format
    print("\n=== Additional Analysis ===")
    
    # Contact body analysis
    if 'body1_name' in df.columns and 'body2_name' in df.columns:
        print(f"Unique body1 names: {df['body1_name'].nunique()}")
        print(f"Unique body2 names: {df['body2_name'].nunique()}")
        print(f"Most common body1: {df['body1_name'].mode().iloc[0] if not df['body1_name'].mode().empty else 'N/A'}")
        print(f"Most common body2: {df['body2_name'].mode().iloc[0] if not df['body2_name'].mode().empty else 'N/A'}")
    
    # Force analysis
    if 'force_magnitude' in df.columns and 'force_normal' in df.columns:
        print(f"\nForce Analysis:")
        print(f"  Total force magnitude: {df['force_magnitude'].sum():.3f} N")
        print(f"  Total normal force: {df['force_normal'].sum():.3f} N")
        print(f"  Average force per contact: {df['force_magnitude'].mean():.3f} N")
        print(f"  Average normal force per contact: {df['force_normal'].mean():.3f} N")
        
        # Contact quality analysis
        if 'force_normal' in df.columns:
            positive_normal = (df['force_normal'] > 0).sum()
            total_contacts = len(df)
            print(f"  Contact quality: {positive_normal}/{total_contacts} ({100*positive_normal/total_contacts:.1f}% positive normal forces)")
    
    print("=" * 30)

def show_available_joints(df):
    """Show available joint/body names in the data"""
    print("\n=== Available Joint/Body Names ===")
    
    # Get unique body names
    body1_names = df['body1_name'].unique() if 'body1_name' in df.columns else []
    body2_names = df['body2_name'].unique() if 'body2_name' in df.columns else []
    all_bodies = sorted(set(list(body1_names) + list(body2_names)))
    
    print(f"Found {len(all_bodies)} unique body/joint names:")
    for i, body_name in enumerate(all_bodies, 1):
        print(f"  {i:2d}. {body_name}")
    
    return all_bodies

def parse_joint_filter(joint_filter_str, available_bodies):
    """Parse joint filter string from command line"""
    if not joint_filter_str:
        return None
    
    try:
        if joint_filter_str.lower() == 'all':
            return available_bodies
        
        if joint_filter_str.lower() == 'none':
            return []
        
        # Parse user input
        selected_indices = []
        for part in joint_filter_str.split(','):
            part = part.strip()
            if '-' in part:
                # Handle ranges like "1-5"
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start, end + 1))
            else:
                selected_indices.append(int(part))
        
        # Convert indices to body names
        selected_bodies = []
        for idx in selected_indices:
            if 1 <= idx <= len(available_bodies):
                selected_bodies.append(available_bodies[idx - 1])
            else:
                print(f"Warning: Index {idx} is out of range (1-{len(available_bodies)})")
        
        if selected_bodies:
            print(f"Command line filter: Selected {len(selected_bodies)} joints to exclude:")
            for body in selected_bodies:
                print(f"  - {body}")
            return selected_bodies
        else:
            print("No valid joints selected from command line filter.")
            return None
            
    except ValueError:
        print("Error: Invalid joint filter format. Use numbers separated by commas, or 'all'/'none'")
        return None

def get_user_joint_selection(available_bodies, joint_filter_str=None):
    """Get user selection for which joints to filter out"""
    # If joint filter is provided via command line, use it
    if joint_filter_str:
        parsed_filter = parse_joint_filter(joint_filter_str, available_bodies)
        if parsed_filter is not None:
            return parsed_filter
        # If parsing failed, fall back to interactive mode
        print("Falling back to interactive mode...")
    
    print("\n=== Joint Filter Selection ===")
    print("You can choose to filter out specific joints/bodies from the visualization.")
    print("Enter the numbers of joints you want to EXCLUDE (separated by commas), or press Enter to skip:")
    print("Example: 1,3,5,7 (to exclude joints 1, 3, 5, and 7)")
    print("Example: 1-5 (to exclude joints 1 through 5)")
    print("Example: all (to exclude all joints)")
    print("Example: none (to include all joints)")
    print("Press Enter to use default foot filtering only")
    
    while True:
        try:
            user_input = input("Your selection: ").strip()
            
            if not user_input:
                # Default: only filter foot contacts
                return ['LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R', 'LINK_ANKLE_PITCH_L', 'LINK_ANKLE_PITCH_R']
            
            if user_input.lower() == 'all':
                return available_bodies
            
            if user_input.lower() == 'none':
                return []
            
            # Parse user input
            selected_indices = []
            for part in user_input.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "1-5"
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    selected_indices.append(int(part))
            
            # Convert indices to body names
            selected_bodies = []
            for idx in selected_indices:
                if 1 <= idx <= len(available_bodies):
                    selected_bodies.append(available_bodies[idx - 1])
                else:
                    print(f"Warning: Index {idx} is out of range (1-{len(available_bodies)})")
            
            if selected_bodies:
                print(f"Selected {len(selected_bodies)} joints to exclude:")
                for body in selected_bodies:
                    print(f"  - {body}")
                return selected_bodies
            else:
                print("No valid joints selected. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or 'all'/'none'")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return ['LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R', 'LINK_ANKLE_PITCH_L', 'LINK_ANKLE_PITCH_R']

def cluster_nearby_contacts(contact_forces, min_distance=0.1):
    """Cluster nearby contact points to avoid overlapping spheres
    min_distance: minimum distance between sphere centers in meters
    Uses proper clustering algorithm that groups all points within distance threshold"""
    if not contact_forces:
        return []
    
    print(f"Clustering nearby contacts with minimum distance: {min_distance}m")
    
    # For very large datasets, use simpler clustering
    if len(contact_forces) > 10000:
        print("Large dataset detected, using simplified clustering...")
        return cluster_nearby_contacts_simple(contact_forces, min_distance)
    
    # Convert to numpy arrays for easier computation
    positions = np.array([cf['position'] for cf in contact_forces])
    forces = [cf['max_force'] for cf in contact_forces]
    
    # Initialize clusters - each point starts as its own cluster
    n_points = len(contact_forces)
    cluster_ids = list(range(n_points))
    
    # Union-Find data structure for clustering
    def find_root(x):
        # Use iterative approach to avoid recursion depth issues
        while cluster_ids[x] != x:
            cluster_ids[x] = cluster_ids[cluster_ids[x]]  # Path compression
            x = cluster_ids[x]
        return x
    
    def union(x, y):
        root_x, root_y = find_root(x), find_root(y)
        if root_x != root_y:
            cluster_ids[root_x] = root_y
    
    # Build distance matrix and merge nearby clusters
    # Limit the number of comparisons for large datasets
    max_comparisons = min(1000000, n_points * (n_points - 1) // 2)  # Limit to 1M comparisons
    comparison_count = 0
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if comparison_count >= max_comparisons:
                print(f"Warning: Reached maximum comparisons limit ({max_comparisons}), stopping clustering")
                break
                
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < min_distance:
                union(i, j)
            comparison_count += 1
        
        if comparison_count >= max_comparisons:
            break
    
    # Group points by cluster
    clusters = {}
    for i in range(n_points):
        root = find_root(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    # For each cluster, select the contact with maximum force
    clustered_forces = []
    for cluster_indices in clusters.values():
        cluster_contacts = [contact_forces[i] for i in cluster_indices]
        max_force_cf = max(cluster_contacts, key=lambda x: x['max_force'])
        clustered_forces.append(max_force_cf)
    
    print(f"Clustered {len(contact_forces)} contacts into {len(clustered_forces)} non-overlapping spheres")
    
    # Debug: Show clustering statistics
    if len(contact_forces) > 0:
        reduction_ratio = len(clustered_forces) / len(contact_forces)
        print(f"Clustering reduction: {reduction_ratio:.1%} ({len(contact_forces)} -> {len(clustered_forces)})")
        
        if reduction_ratio < 0.1:  # Less than 10% remaining
            print("Warning: Heavy clustering detected. Consider increasing min_distance for more points.")
        elif reduction_ratio > 0.8:  # More than 80% remaining
            print("Info: Light clustering. Consider decreasing min_distance to reduce overlaps.")
    
    return clustered_forces

def cluster_nearby_contacts_simple(contact_forces, min_distance=0.1):
    """Simplified clustering for large datasets using spatial hashing"""
    if not contact_forces:
        return []
    
    print(f"Using simplified clustering for large dataset with minimum distance: {min_distance}m")
    
    # Use spatial hashing for faster clustering
    hash_size = min_distance
    spatial_hash = {}
    
    # Hash all points
    for i, cf in enumerate(contact_forces):
        pos = cf['position']
        hash_key = (int(pos[0] / hash_size), int(pos[1] / hash_size), int(pos[2] / hash_size))
        if hash_key not in spatial_hash:
            spatial_hash[hash_key] = []
        spatial_hash[hash_key].append((i, cf))
    
    # Cluster points within the same or adjacent hash cells
    clustered_forces = []
    processed = set()
    
    for hash_key, points in spatial_hash.items():
        if len(points) == 0:
            continue
            
        # Find the point with maximum force in this cell
        max_force_cf = max(points, key=lambda x: x[1]['max_force'])
        clustered_forces.append(max_force_cf[1])
        processed.add(max_force_cf[0])
        
        # Check adjacent cells for nearby points
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    adj_key = (hash_key[0] + dx, hash_key[1] + dy, hash_key[2] + dz)
                    if adj_key in spatial_hash:
                        for i, cf in spatial_hash[adj_key]:
                            if i not in processed:
                                # Check if within distance
                                pos1 = max_force_cf[1]['position']
                                pos2 = cf['position']
                                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                                if distance < min_distance:
                                    processed.add(i)
    
    print(f"Simplified clustering: {len(contact_forces)} -> {len(clustered_forces)} points")
    return clustered_forces

def create_contact_force_visualization(df, x_col, y_col, z_col, max_spheres_override=None, custom_filter_bodies=None, min_points_per_link=5, min_sphere_distance=0.01, enable_clustering=True, uniform_distribution=False):
    """Create contact force visualization data with sphere sizes based on force magnitude
    Now supports proportional distribution by body2_name with minimum points per link
    min_sphere_distance: minimum distance between sphere centers to avoid overlap"""
    print("Processing contact force data for sphere visualization...")
    
    # Filter valid coordinates
    valid_mask = (df[x_col].notna() & df[y_col].notna() & df[z_col].notna() & 
                  np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[z_col]))
    
    if not valid_mask.any():
        print("No valid contact coordinates found")
        return []
    
    df_valid = df[valid_mask].copy()
    print(f"Found {len(df_valid)} valid contact points")
    
    # Filter out contacts based on user selection
    if custom_filter_bodies is not None:
        # Use custom filter bodies
        filter_bodies = custom_filter_bodies
        filter_name = "custom joints"
    else:
        filter_bodies = []
        filter_name = "no filtering"
    
    if filter_bodies:
        before_filter = len(df_valid)
        contact_mask = (
            df_valid['body1_name'].isin(filter_bodies) | 
            df_valid['body2_name'].isin(filter_bodies)
        )
        df_valid = df_valid[~contact_mask]
        after_filter = len(df_valid)
        
        print(f"Filtered out {before_filter - after_filter} {filter_name}")
        print(f"Remaining contact points: {after_filter}")
        
        if after_filter == 0:
            print(f"Warning: All contacts were {filter_name} and have been filtered out")
            return []
    
    # Check if body2_name column exists
    if 'body2_name' not in df_valid.columns:
        print("Warning: body2_name column not found, using original method")
        return create_contact_force_visualization_original(df_valid, x_col, y_col, z_col, max_spheres_override)
    
    # Group by body2_name and position to aggregate forces
    print("Grouping contact points by body2_name and position...")
    body_groups = {}
    
    for _, row in df_valid.iterrows():
        body2_name = row['body2_name']
        pos = (row[x_col], row[y_col], row[z_col])
        
        # Use force magnitude if available, otherwise calculate from components
        if 'force_magnitude' in row and pd.notna(row['force_magnitude']):
            force_mag = row['force_magnitude']
        elif all(col in row for col in ['force_x', 'force_y', 'force_z']) and all(pd.notna(row[col]) for col in ['force_x', 'force_y', 'force_z']):
            force_mag = np.sqrt(row['force_x']**2 + row['force_y']**2 + row['force_z']**2)
        elif 'force_normal' in row and pd.notna(row['force_normal']):
            # Use normal force as fallback
            force_mag = abs(row['force_normal'])
        else:
            force_mag = 1.0  # Default force if not available
        
        if body2_name not in body_groups:
            body_groups[body2_name] = {}
        
        if pos in body_groups[body2_name]:
            body_groups[body2_name][pos]['forces'].append(force_mag)
            body_groups[body2_name][pos]['count'] += 1
        else:
            body_groups[body2_name][pos] = {
                'position': pos,
                'forces': [force_mag],
                'count': 1
            }
    
    # Convert to contact forces list for clustering
    all_contact_forces = []
    for body_name, position_groups in body_groups.items():
        for pos, data in position_groups.items():
            max_force = max(data['forces'])
            all_contact_forces.append({
                'body2_name': body_name,
                'position': data['position'],
                'max_force': max_force,
                'count': data['count']
            })
    
    print(f"Found {len(all_contact_forces)} unique contact positions")
    
    # STEP 1: Apply clustering based on user setting
    if enable_clustering:
        print("Step 1: Clustering nearby contacts...")
        clustered_contact_forces = cluster_nearby_contacts(all_contact_forces, min_sphere_distance)
    else:
        print("Step 1: Using all unique contact positions (clustering disabled)")
        clustered_contact_forces = all_contact_forces  # 直接使用所有唯一位置
    
    # STEP 2: Group results by body2_name for proportional distribution
    body_contact_forces = {}
    for cf in clustered_contact_forces:
        body_name = cf['body2_name']
        if body_name not in body_contact_forces:
            body_contact_forces[body_name] = []
        body_contact_forces[body_name].append(cf)
    
    # Sort each body's contacts by force magnitude (highest first)
    for body_name in body_contact_forces:
        body_contact_forces[body_name].sort(key=lambda x: x['max_force'], reverse=True)
    
    # Calculate total contact points and determine max spheres
    total_contacts = sum(len(forces) for forces in body_contact_forces.values())
    num_bodies = len(body_contact_forces)
    
    if max_spheres_override is not None:
        max_spheres = max_spheres_override
        print(f"Using user-specified max spheres: {max_spheres}")
    else:
        # Smart limit based on data size
        if total_contacts > 10000:
            max_spheres = 2000
            print(f"Very large dataset detected ({total_contacts} points), limiting to top {max_spheres} points")
        elif total_contacts > 5000:
            max_spheres = 1000
            print(f"Large dataset detected ({total_contacts} points), limiting to top {max_spheres} points")
        elif total_contacts > 2000:
            max_spheres = 500
            print(f"Medium dataset detected ({total_contacts} points), limiting to top {max_spheres} points")
        elif total_contacts > 500:
            max_spheres = 200
            print(f"Small dataset detected ({total_contacts} points), limiting to top {max_spheres} points")
        else:
            max_spheres = total_contacts
            print(f"Small dataset ({total_contacts} points), displaying all contact points")
    
    # Calculate distribution based on uniform_distribution parameter
    if uniform_distribution:
        print(f"\n=== Step 2: 按body2_name均匀分配显示点 ===")
        print(f"分配方式: 均匀分配（每个link获得相同数量的额外点）")
    else:
        print(f"\n=== Step 2: 按body2_name比例分配显示点 ===")
        print(f"分配方式: 比例分配（接触点多的link获得更多显示点）")
    
    print(f"聚类后接触点数: {total_contacts}")
    print(f"总body数量: {num_bodies}")
    print(f"最大显示点数: {max_spheres}")
    print(f"每个body最少显示点数: {min_points_per_link}")
    
    # Calculate points per body
    body_point_counts = {}
    remaining_points = max_spheres
    
    # First pass: assign minimum points to each body
    for body_name in body_contact_forces.keys():
        body_point_counts[body_name] = min_points_per_link
        remaining_points -= min_points_per_link
    
    # If we don't have enough points for minimum, adjust
    if remaining_points < 0:
        print(f"Warning: Not enough points for minimum per body. Adjusting minimum to {max_spheres // num_bodies}")
        min_points_per_link = max_spheres // num_bodies
        for body_name in body_contact_forces.keys():
            body_point_counts[body_name] = min_points_per_link
        remaining_points = max_spheres - (min_points_per_link * num_bodies)
    
    # Second pass: distribute remaining points based on distribution type
    if remaining_points > 0:
        if uniform_distribution:
            # Uniform distribution: each body gets equal additional points
            points_per_body = remaining_points // num_bodies
            extra_points = remaining_points % num_bodies
            
            for i, body_name in enumerate(body_contact_forces.keys()):
                additional_points = points_per_body + (1 if i < extra_points else 0)
                body_point_counts[body_name] += additional_points
        else:
            # Proportional distribution: bodies with more contacts get more points
            total_available = sum(max(0, len(body_contact_forces[body_name]) - min_points_per_link) 
                                for body_name in body_contact_forces.keys())
            
            if total_available > 0:
                for body_name, contact_forces in body_contact_forces.items():
                    available_points = max(0, len(contact_forces) - min_points_per_link)
                    if available_points > 0:
                        # Proportional distribution
                        additional_points = int(remaining_points * available_points / total_available)
                        body_point_counts[body_name] += additional_points
    
    # Final pass: ensure we don't exceed available points for each body
    for body_name, contact_forces in body_contact_forces.items():
        body_point_counts[body_name] = min(body_point_counts[body_name], len(contact_forces))
    
    # Redistribute points from bodies with no contacts to bodies with contacts
    bodies_with_contacts = [body_name for body_name, contact_forces in body_contact_forces.items() if len(contact_forces) > 0]
    bodies_without_contacts = [body_name for body_name, contact_forces in body_contact_forces.items() if len(contact_forces) == 0]
    
    if bodies_without_contacts and bodies_with_contacts:
        print(f"\n=== 重新分配无碰撞点的显示量 ===")
        print(f"无碰撞点的body: {bodies_without_contacts}")
        print(f"有碰撞点的body: {bodies_with_contacts}")
        
        # Calculate total points to redistribute
        points_to_redistribute = sum(body_point_counts[body_name] for body_name in bodies_without_contacts)
        print(f"需要重新分配的显示点: {points_to_redistribute}")
        
        if points_to_redistribute > 0:
            # Reset points for bodies without contacts
            for body_name in bodies_without_contacts:
                body_point_counts[body_name] = 0
            
            # Redistribute points to bodies with contacts
            if uniform_distribution:
                # Uniform redistribution
                points_per_body = points_to_redistribute // len(bodies_with_contacts)
                extra_points = points_to_redistribute % len(bodies_with_contacts)
                
                for i, body_name in enumerate(bodies_with_contacts):
                    additional_points = points_per_body + (1 if i < extra_points else 0)
                    body_point_counts[body_name] += additional_points
            else:
                # Proportional redistribution based on contact count
                total_contacts = sum(len(body_contact_forces[body_name]) for body_name in bodies_with_contacts)
                if total_contacts > 0:
                    for body_name in bodies_with_contacts:
                        contact_count = len(body_contact_forces[body_name])
                        additional_points = int(points_to_redistribute * contact_count / total_contacts)
                        body_point_counts[body_name] += additional_points
            
            print(f"重新分配完成，各body新的显示点数:")
            for body_name in bodies_with_contacts:
                print(f"  {body_name}: {body_point_counts[body_name]} 点")
    
    # Select contact forces for each body
    final_contact_forces = []
    total_selected = 0
    
    print(f"\n=== 各body分配结果 ===")
    for body_name, contact_forces in body_contact_forces.items():
        num_to_select = body_point_counts[body_name]
        selected_forces = contact_forces[:num_to_select]
        final_contact_forces.extend(selected_forces)
        total_selected += num_to_select
        
        if selected_forces:
            max_force = max(cf['max_force'] for cf in selected_forces)
            min_force = min(cf['max_force'] for cf in selected_forces)
            print(f"  {body_name}: {num_to_select}/{len(contact_forces)} 点, 力范围: {min_force:.3f}-{max_force:.3f} N")
        else:
            print(f"  {body_name}: 0/{len(contact_forces)} 点")
    
    print(f"\n总计选择: {total_selected} 个接触点")
    
    # Sort final results by force magnitude for consistent visualization
    final_contact_forces.sort(key=lambda x: x['max_force'], reverse=True)
    
    # Clustering already applied in Step 1, no need to cluster again
    
    print(f"Created {len(final_contact_forces)} contact points for visualization")
    if final_contact_forces:
        max_force = max(cf['max_force'] for cf in final_contact_forces)
        min_force = min(cf['max_force'] for cf in final_contact_forces)
        print(f"Overall force range: {min_force:.3f} - {max_force:.3f} N")
        
        # Debug: Show force distribution
        forces = [cf['max_force'] for cf in final_contact_forces]
        print(f"Force statistics: mean={np.mean(forces):.3f}, std={np.std(forces):.3f}, median={np.median(forces):.3f}")
    
    return final_contact_forces

def create_contact_force_visualization_original(df_valid, x_col, y_col, z_col, max_spheres_override=None):
    """Original contact force visualization method (fallback)"""
    print("Using original contact force visualization method...")
    
    # Group by position to aggregate forces at the same location
    position_groups = {}
    
    for _, row in df_valid.iterrows():
        pos = (row[x_col], row[y_col], row[z_col])
        
        # Use force magnitude if available, otherwise calculate from components
        if 'force_magnitude' in row and pd.notna(row['force_magnitude']):
            force_mag = row['force_magnitude']
        elif all(col in row for col in ['force_x', 'force_y', 'force_z']) and all(pd.notna(row[col]) for col in ['force_x', 'force_y', 'force_z']):
            force_mag = np.sqrt(row['force_x']**2 + row['force_y']**2 + row['force_z']**2)
        elif 'force_normal' in row and pd.notna(row['force_normal']):
            # Use normal force as fallback
            force_mag = abs(row['force_normal'])
        else:
            force_mag = 1.0  # Default force if not available
        
        if pos in position_groups:
            position_groups[pos]['forces'].append(force_mag)
            position_groups[pos]['count'] += 1
        else:
            position_groups[pos] = {
                'position': pos,
                'forces': [force_mag],
                'count': 1
            }
    
    # Convert to list and calculate max force for each position
    contact_forces = []
    for pos, data in position_groups.items():
        max_force = max(data['forces'])
        contact_forces.append({
            'position': data['position'],
            'max_force': max_force,
            'count': data['count']
        })
    
    # Sort by force magnitude (highest first)
    contact_forces.sort(key=lambda x: x['max_force'], reverse=True)
    
    # Apply limits
    if max_spheres_override is not None:
        max_spheres = max_spheres_override
        if len(contact_forces) > max_spheres:
            contact_forces = contact_forces[:max_spheres]
    else:
        # Smart limit based on data size
        if len(contact_forces) > 10000:
            max_spheres = 2000
        elif len(contact_forces) > 5000:
            max_spheres = 1000
        elif len(contact_forces) > 2000:
            max_spheres = 500
        elif len(contact_forces) > 500:
            max_spheres = 200
        else:
            max_spheres = len(contact_forces)
        
        if len(contact_forces) > max_spheres:
            contact_forces = contact_forces[:max_spheres]
    
    return contact_forces

def add_contact_spheres_to_xml(xml_path, contact_forces):
    """Add contact force spheres to XML file"""
    print(f"Adding {len(contact_forces)} contact force spheres to XML...")
    
    # Read original XML
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    
    # Find the end of worldbody section
    worldbody_end = xml_content.find('</worldbody>')
    if worldbody_end == -1:
        print("Warning: Could not find </worldbody> tag, adding spheres at end")
        worldbody_end = xml_content.find('</mujoco>')
        if worldbody_end == -1:
            print("Error: Could not find </mujoco> tag")
            return None
    
    # Prepare contact spheres XML
    spheres_xml = '\n    <!-- Contact Force Spheres -->\n'
    max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
    
    for i, cf in enumerate(contact_forces):
        # Calculate sphere radius based on force magnitude
        # Base radius: 0.01, max additional radius: 0.04
        radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
        pos = cf['position']
        force = cf['max_force']
        
        # Color based on force magnitude (red to yellow)
        force_ratio = cf['max_force'] / max_force if max_force > 0 else 0
        r = 1.0
        g = force_ratio
        b = 0.0
        a = 0.8
        
        spheres_xml += f'''    <body name="contact_sphere_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}" group="1"/>
      <site name="contact_site_{i}" pos="0 0 0" size="0.001"/>
    </body>
'''
    
    # Insert spheres before </worldbody>
    xml_content = xml_content[:worldbody_end] + spheres_xml + xml_content[worldbody_end:]
    
    # Create temporary XML file with contact spheres
    xml_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    xml_temp.write(xml_content)
    xml_temp.close()
    
    return xml_temp.name

def show_mujoco_viewer_with_spheres(model, data, contact_forces):
    """Show Mujoco viewer with contact force spheres"""
    print("Starting Mujoco viewer with contact force spheres...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Right click: Pan")
    print("  - ESC: Exit")
    print("  - 在左侧Rendering面板中，将Label设置为'Force'")
    print("  - 在Model Elements中，点击'Contact Force'按钮（会显示红色边框）")
    print("  - 这样就会在3D视图中显示接触力的数值标签")
    print("  - 按 'L' 键也可以切换标签显示")
    print("  - 当前已启用接触力数值标签显示")
    
    # Import viewer module
    try:
        from mujoco import viewer
    except ImportError:
        try:
            import mujoco.viewer as viewer
        except ImportError:
            print("Error: mujoco.viewer module not available")
            print("Please install mujoco with viewer support")
            return
    
    # Set up the viewer
    with viewer.launch_passive(model, data) as viewer_handle:
        # Set initial camera position
        viewer_handle.cam.distance = 3.0
        viewer_handle.cam.azimuth = 45
        viewer_handle.cam.elevation = -20
        
        # Enable contact force visualization
        viewer_handle.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # 显示接触点
        
        # Add contact force information to viewer
        if contact_forces:
            max_force = max(cf['max_force'] for cf in contact_forces)
            print(f"\nContact Force Visualization:")
            print(f"Total contact points: {len(contact_forces)}")
            print(f"Max force: {max_force:.3f} N")
            print(f"Sphere sizes represent force magnitude")
            print(f"Colors: Red (high force) to Yellow (low force)")
            print(f"Contact points are enabled")
            print(f"Contact force labels are enabled")
            print(f"To see force values in 3D viewer:")
            print(f"  1. Set Label to 'Force' in left panel")
            print(f"  2. Click 'Contact Force' button in Model Elements")
            print(f"  3. Force values will appear as white text labels")
            print(f"Force values are also printed to console")
            
            # Print top 10 highest force values for reference
            print(f"\nTop 10 Highest Force Values:")
            sorted_forces = sorted(contact_forces, key=lambda x: x['max_force'], reverse=True)
            for i, cf in enumerate(sorted_forces[:10]):
                pos = cf['position']
                force = cf['max_force']
                print(f"  {i+1:2d}. {force:8.3f} N at ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
        
        # Keep viewer open
        while viewer_handle.is_running():
            time.sleep(0.01)

def merge_xml_includes(xml_content, base_dir):
    """Merge all include files into a single XML content"""
    print(f"Merging XML includes in directory: {base_dir}")
    
    try:
        root = ET.fromstring(xml_content)
        
        # Find all include elements
        includes = root.findall('.//include')
        
        for include_elem in includes:
            file_attr = include_elem.get('file')
            if file_attr:
                # Convert relative path to absolute path
                abs_include_path = os.path.join(base_dir, file_attr)
                print(f"  Merging include file: {file_attr} -> {abs_include_path}")
                
                # Check if the included file exists
                if os.path.exists(abs_include_path):
                    # Read the included file
                    with open(abs_include_path, 'r') as f:
                        included_content = f.read()
                    
                    # Recursively merge the included file
                    included_dir = os.path.dirname(abs_include_path)
                    merged_content = merge_xml_includes(included_content, included_dir)
                    
                    # Parse the merged content
                    included_root = ET.fromstring(merged_content)
                    
                    # Find the parent of the include element by searching
                    for parent in root.iter():
                        if include_elem in list(parent):
                            # Remove the include element
                            parent.remove(include_elem)
                            
                            # Insert all children of the included root
                            for child in included_root:
                                parent.append(child)
                            break
                    else:
                        # If no parent found, just remove the include
                        root.remove(include_elem)
                else:
                    print(f"  Warning: Included file not found: {abs_include_path}")
        
        # Process mesh elements in assets
        for mesh_elem in root.findall('.//mesh'):
            file_attr = mesh_elem.get('file')
            if file_attr:
                # Convert relative path to absolute path
                abs_mesh_path = os.path.join(base_dir, file_attr)
                print(f"  Mesh file: {file_attr} -> {abs_mesh_path}")
                mesh_elem.set('file', abs_mesh_path)
        
        # Convert back to string
        return ET.tostring(root, encoding='unicode')
        
    except ET.ParseError as e:
        print(f"Warning: Could not parse XML for include processing: {e}")
        return xml_content

def adjust_memory_for_spheres(num_spheres):
    """Adjust memory allocation based on number of spheres"""
    if num_spheres > 1500:
        # Very high memory usage - increase limits significantly
        os.environ['MUJOCO_STACK_SIZE'] = '536870912'  # 512MB stack (doubled for 2000+ spheres)
        os.environ['MUJOCO_ARENA_SIZE'] = '4294967296'  # 4GB arena (doubled for 2000+ spheres)
        print(f"Very high memory mode: {num_spheres} spheres -> 512MB stack, 4GB arena")
    elif num_spheres > 1000:
        # High memory usage
        os.environ['MUJOCO_STACK_SIZE'] = '268435456'  # 256MB stack
        os.environ['MUJOCO_ARENA_SIZE'] = '2147483648'  # 2GB arena
        print(f"High memory mode: {num_spheres} spheres -> 256MB stack, 2GB arena")
    elif num_spheres > 500:
        # Medium memory usage
        os.environ['MUJOCO_STACK_SIZE'] = '134217728'   # 128MB stack
        os.environ['MUJOCO_ARENA_SIZE'] = '1073741824'   # 1GB arena
        print(f"Medium memory mode: {num_spheres} spheres -> 128MB stack, 1GB arena")
    else:
        # Normal memory usage
        os.environ['MUJOCO_STACK_SIZE'] = '67108864'    # 64MB stack
        os.environ['MUJOCO_ARENA_SIZE'] = '536870912'   # 512MB arena
        print(f"Normal memory mode: {num_spheres} spheres -> 64MB stack, 512MB arena")

def load_mujoco_model_with_contact_spheres(xml_file, contact_forces):
    """Load Mujoco model with contact spheres, handling relative paths correctly"""
    print(f"Loading Mujoco model from: {xml_file}")
    
    # Print current memory settings
    print(f"Current MUJOCO_STACK_SIZE: {os.environ.get('MUJOCO_STACK_SIZE', 'Not set')}")
    print(f"Current MUJOCO_ARENA_SIZE: {os.environ.get('MUJOCO_ARENA_SIZE', 'Not set')}")
    
    # Adjust memory allocation based on number of spheres
    if contact_forces:
        adjust_memory_for_spheres(len(contact_forces))
    
    # Get the absolute path of the XML file
    xml_abs_path = os.path.abspath(xml_file)
    xml_dir = os.path.dirname(xml_abs_path)
    
    # Read the original XML file
    with open(xml_abs_path, 'r') as f:
        xml_content = f.read()
    
    # Merge all includes into a single XML content
    xml_content = merge_xml_includes(xml_content, xml_dir)
    
    # Add contact spheres if available
    if contact_forces:
        # Find the end of worldbody section
        worldbody_end = xml_content.find('</worldbody>')
        if worldbody_end == -1:
            print("Warning: Could not find </worldbody> tag, adding spheres at end")
            worldbody_end = xml_content.find('</mujoco>')
            if worldbody_end == -1:
                print("Error: Could not find </mujoco> tag")
                return None, None, None
        
        # Prepare contact spheres XML
        spheres_xml = '\n    <!-- Contact Force Spheres -->\n'
        max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
        
        print(f"Adding {len(contact_forces)} contact force spheres...")
        print(f"Max force: {max_force:.3f} N")
        
        for i, cf in enumerate(contact_forces):
            # Calculate sphere radius based on force magnitude (improved scaling)
            force_ratio = cf['max_force'] / max_force if max_force > 0 else 0
            # Use logarithmic scaling for better visualization
            log_ratio = np.log10(1 + 9 * force_ratio) / np.log10(10)  # Maps 0-1 to 0-1 with log scaling
            radius = 0.003 + 0.01 * log_ratio  # Range: 0.003 - 0.01 meters (3-10mm)
            pos = cf['position']
            
            # Color based on force magnitude (red for high force, yellow for low force)
            r = 1.0
            g = 0.2 + 0.8 * (1.0 - force_ratio)  # Range: 1.0 - 0.2 (yellow to red)
            b = 0.0
            a = 0.8
            
            # Debug: Print sphere info for first few spheres
            if i < 5:
                print(f"  Sphere {i}: force={cf['max_force']:.3f}N, ratio={force_ratio:.3f}, radius={radius:.4f}m, color=({r:.2f},{g:.2f},{b:.2f})")
            
            spheres_xml += f'''    <body name="contact_sphere_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}" group="1"/>
    </body>
'''
        
        # Insert spheres before </worldbody>
        xml_content = xml_content[:worldbody_end] + spheres_xml + xml_content[worldbody_end:]
    
    # Create temporary XML file
    xml_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    xml_temp.write(xml_content)
    xml_temp.close()
    
    print(f"Created temporary XML file: {xml_temp.name}")
    
    # Load Mujoco model from temporary file
    try:
        model = mj.MjModel.from_xml_path(xml_temp.name)
        data = mj.MjData(model)
        
        # Set MuJoCo memory options for better performance with many spheres
        # Note: These options may not be available in all MuJoCo versions
        try:
            # Increase memory allocation for large models with many constraints
            mj.mj_setOption(model, mj.mjtOption.mjOptionStack, 0)  # Use larger stack
            mj.mj_setOption(model, mj.mjtOption.mjOptionMemory, 0)  # Use larger memory
            # Set arena size for constraints (if available)
            if hasattr(mj, 'mj_setOption'):
                mj.mj_setOption(model, mj.mjtOption.mjOptionArena, 0)  # Use larger arena
        except AttributeError:
            print("Note: MuJoCo memory optimization options not available in this version")
        
        print("Successfully loaded Mujoco model")
        
        # Apply keyframe pose if available
        if model.nkey > 0:
            print("Applying keyframe pose to robot...")
            # Use the first keyframe (usually the homing pose)
            key_id = 0
            mj.mj_resetDataKeyframe(model, data, key_id)
            print(f"Applied keyframe {key_id} pose")
            
            # Print some joint information for debugging
            if hasattr(model, 'jnt_name'):
                print("Joint names found in model:")
                for i in range(min(10, model.njnt)):  # Print first 10 joints
                    print(f"  Joint {i}: {model.jnt_name[i]}")
        
        return model, data, xml_temp.name
    except Exception as e:
        print(f"Error loading Mujoco model: {e}")
        # Clean up temporary file
        os.unlink(xml_temp.name)
        return None, None, None

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python3 mujoco_xml_contact_display.py <csv_file> <xml_file> [world|robot_frame] [max_spheres] [joint_filter] [enable_clustering] [uniform_distribution]")
        print("")
        print("Examples:")
        print("  # Using robot frame coordinates (default)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml")
        print("")
        print("  # Using world coordinates")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml world")
        print("")
        print("  # Limit to 100 spheres for memory optimization")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 100")
        print("")
        print("  # Display 2000 spheres (high memory usage)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 2000")
        print("")
        print("  # Filter specific joints by numbers (e.g., exclude joints 1,2)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 500 1,2")
        print("")
        print("  # Filter specific joints by range (e.g., exclude joints 1-5)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 500 1-5")
        print("")
        print("  # Enable clustering to reduce overlapping spheres")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 \"\" true")
        print("")
        print("  # Disable clustering to show more points (default)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 \"\" false")
        print("")
        print("  # Enable uniform distribution (each link gets equal points)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 \"\" false true")
        print("")
        print("  # Use proportional distribution (default)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 \"\" false false")
        print("")
        print("CSV Format Support:")
        print("  - New format: timestamp, contact_id, body1_name, body2_name, pos_x/y/z, robot_frame_x/y/z, force_magnitude, force_normal, etc.")
        print("  - Includes: base_link pose, collision_link pose, joint angles")
        print("")
        print("Arguments:")
        print("  csv_file: Path to the CSV file with contact force data")
        print("  xml_file: Path to XML model file (e.g., pm_v2_mesh.xml)")
        print("  world|robot_frame: Coordinate system (default: robot_frame)")
        print("  max_spheres: Maximum number of spheres to display (default: auto)")
        print("  filter_foot: Filter out foot contacts (true/false, default: true)")
        print("  joint_filter: Joint numbers to exclude (e.g., 1,2,3 or 1-5, default: interactive)")
        print("  enable_clustering: Enable contact clustering (true/false, default: false)")
        print("  uniform_distribution: Use uniform distribution (true/false, default: false)")
        print("")
        print("Note: Contact forces will be displayed as spheres with sizes proportional to force magnitude")
        print("Foot contacts (LINK_ANKLE_*) are filtered out by default")
        print("")
        print("New Feature - Proportional Distribution by body2_name:")
        print("  - Contact points are now distributed proportionally across different body2_name links")
        print("  - Each link gets a minimum of 5 points to ensure visibility")
        print("  - Remaining points are distributed based on contact point density per link")
        print("  - This ensures all links are visible, not just the highest force areas")
        print("")
        print("New Feature - Contact Clustering:")
        print("  - Nearby contact points are clustered to avoid overlapping spheres")
        print("  - Each cluster uses the contact with maximum force to represent the area")
        print("  - This prevents visual occlusion and ensures all important force areas are visible")
        print("  - Clustering distance can be adjusted in main() function (min_sphere_distance)")
        print("")
        print("Memory Usage Guidelines:")
        print("  - 100-500 spheres: Normal memory usage")
        print("  - 500-1000 spheres: Moderate memory usage")
        print("  - 1000-2000 spheres: High memory usage (64MB stack, 512MB arena)")
        print("  - 2000+ spheres: Very high memory usage (may cause performance issues)")
        return
    
    csv_file = sys.argv[1]
    xml_file = sys.argv[2]
    
    # Check for coordinate type
    coord_type = "robot_frame"    # default to robot frame coordinates
    max_spheres_override = None
    filter_foot_contacts = True    # default to filter foot contacts
    
    if len(sys.argv) >= 4:
        # Check if the third argument is a number (max_spheres) or coordinate type
        try:
            max_spheres_override = int(sys.argv[3])
            coord_type = "robot_frame"  # default coordinate type
        except ValueError:
            coord_type = sys.argv[3].lower()
            if coord_type not in ["world", "robot_frame"]:
                print("Error: Coordinate type must be 'world' or 'robot_frame'")
                return
    
    # Check for max_spheres parameter
    if len(sys.argv) >= 5:
        try:
            max_spheres_override = int(sys.argv[4])
        except ValueError:
            print("Error: max_spheres must be a number")
            return
    
    # Check for joint_filter parameter
    joint_filter_str = None
    if len(sys.argv) >= 6:
        joint_filter_str = sys.argv[5]
    
    # Check for enable_clustering parameter
    enable_clustering = False  # default to disabled
    if len(sys.argv) >= 7:
        clustering_str = sys.argv[6].lower()
        if clustering_str in ['true', '1', 'yes', 'y']:
            enable_clustering = True
        elif clustering_str in ['false', '0', 'no', 'n']:
            enable_clustering = False
        else:
            print("Error: enable_clustering must be true/false")
            return
    
    # Check for uniform_distribution parameter
    uniform_distribution = False  # default to proportional distribution
    if len(sys.argv) >= 8:
        uniform_str = sys.argv[7].lower()
        if uniform_str in ['true', '1', 'yes', 'y']:
            uniform_distribution = True
        elif uniform_str in ['false', '0', 'no', 'n']:
            uniform_distribution = False
        else:
            print("Error: uniform_distribution must be true/false")
            return
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    if not os.path.exists(xml_file):
        print(f"XML file not found: {xml_file}")
        return
    
    # Load contact data
    print(f"Loading contact data: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Analyze CSV data
    analyze_csv_data(df)
    
    # Show available joints and get user selection
    available_bodies = show_available_joints(df)
    
    # 基础排除的link（写死的）
    base_excluded_links = ['LINK_ANKLE_PITCH_L', 'LINK_ANKLE_PITCH_R', 'LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R', 'LINK_HEAD_YAW']
    
    # 处理用户指定的关节过滤
    if joint_filter_str:
        print(f"\n=== 处理关节过滤参数 ===")
        print(f"用户指定过滤: {joint_filter_str}")
        user_excluded_links = parse_joint_filter(joint_filter_str, available_bodies)
        if user_excluded_links:
            excluded_links = base_excluded_links + user_excluded_links
            print(f"额外排除的关节: {user_excluded_links}")
        else:
            excluded_links = base_excluded_links
    else:
        excluded_links = base_excluded_links
        print(f"\n=== 使用默认关节过滤 ===")
    
    print(f"总共排除的关节: {excluded_links}")
    
    # 聚类参数设置
    min_sphere_distance = 0.005  # 最小球体中心距离 (米) - 如果两个点距离 < min_sphere_distance，就合并为一类
    print(f"\n=== 聚类参数设置 ===")
    print(f"聚类功能: {'启用' if enable_clustering else '禁用'}")
    if enable_clustering:
        print(f"最小球体距离: {min_sphere_distance}m ({min_sphere_distance*1000:.1f}mm)")
        print("提示: 减小此值可显示更多点，增大此值可减少重叠")
    else:
        print("提示: 聚类已禁用，将显示所有唯一位置点")
    
    # 打印所有可用的link
    print(f"\n=== 所有可用的Link名称 ===")
    for i, body in enumerate(available_bodies, 1):
        print(f"  {i:2d}. {body}")
    
    # 打印要排除的link
    print(f"\n=== 排除的Link名称 ===")
    for i, link in enumerate(excluded_links, 1):
        print(f"  {i}. {link}")
    
    # 使用预设的排除列表
    custom_filter_bodies = excluded_links
    
    # Check for coordinate columns
    if coord_type == "world":
        # Use world coordinates
        if 'pos_x' in df.columns and 'pos_y' in df.columns and 'pos_z' in df.columns:
            x_col, y_col, z_col = 'pos_x', 'pos_y', 'pos_z'
            print("Using world coordinates (pos_x/y/z)")
        else:
            print("World coordinates (pos_x/y/z) not found in CSV")
            return
    else:
        # Use robot frame coordinates (default)
        if 'robot_frame_x' in df.columns and 'robot_frame_y' in df.columns and 'robot_frame_z' in df.columns:
            x_col, y_col, z_col = 'robot_frame_x', 'robot_frame_y', 'robot_frame_z'
            print("Using robot frame coordinates (robot_frame_x/y/z)")
        else:
            print("Robot frame coordinates (robot_frame_x/y/z) not found in CSV")
            print("Falling back to world coordinates (pos_x/y/z)")
            if 'pos_x' in df.columns and 'pos_y' in df.columns and 'pos_z' in df.columns:
                x_col, y_col, z_col = 'pos_x', 'pos_y', 'pos_z'
                coord_type = "world"
                print("Using world coordinates (pos_x/y/z)")
            else:
                print("No valid coordinate columns found")
                return
    
    # Create contact force visualization with distribution by body2_name
    # Use clustering settings from main function
    contact_forces = create_contact_force_visualization(df, x_col, y_col, z_col, max_spheres_override, excluded_links, min_points_per_link=1, min_sphere_distance=min_sphere_distance, enable_clustering=enable_clustering, uniform_distribution=uniform_distribution)
    
    if not contact_forces:
        print("No contact forces to display")
        return
    
    # 打印显示的力大小范围
    if contact_forces:
        max_force = max(cf['max_force'] for cf in contact_forces)
        min_force = min(cf['max_force'] for cf in contact_forces)
        print(f"\n=== 显示的力大小范围 ===")
        print(f"最小力: {min_force:.3f} N")
        print(f"最大力: {max_force:.3f} N")
        print(f"力范围: {min_force:.3f} - {max_force:.3f} N")
        print(f"力差值: {max_force - min_force:.3f} N")
        
        # 计算力的统计信息
        forces = [cf['max_force'] for cf in contact_forces]
        print(f"平均力: {np.mean(forces):.3f} N")
        print(f"中位数力: {np.median(forces):.3f} N")
        print(f"标准差: {np.std(forces):.3f} N")
        print(f"显示接触点数量: {len(contact_forces)}")
        print("=" * 50)
    
    # Load Mujoco model with contact spheres
    model, data, xml_temp_path = load_mujoco_model_with_contact_spheres(xml_file, contact_forces)
    
    if model is None:
        print("Failed to load Mujoco model")
        return
    
    try:
        # Show interactive viewer with contact spheres
        show_mujoco_viewer_with_spheres(model, data, contact_forces)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary file
        if xml_temp_path and os.path.exists(xml_temp_path):
            os.unlink(xml_temp_path)
    
    print("Viewer closed successfully!")

if __name__ == "__main__":
    main() 