#!/usr/bin/env python3
"""
Mujoco URDF Contact Force Display
Uses official Mujoco Python interface to display URDF model with contact forces
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import mujoco as mj
import tempfile
import xml.etree.ElementTree as ET
import mediapy as media
from datetime import datetime

def fix_urdf_paths(urdf_file_path):
    """Fix URDF file paths by replacing package:// with absolute paths"""
    import tempfile
    
    # Read the original URDF file
    with open(urdf_file_path, 'r') as f:
        urdf_content = f.read()
    
    # Replace package:// paths with absolute paths
    base_path = os.path.dirname(urdf_file_path)  # urdf directory
    robot_path = os.path.dirname(base_path)       # pm_v2 directory
    meshes_path = os.path.join(robot_path, 'meshes')
    meshes_path = os.path.abspath(meshes_path)
    
    print(f"URDF file: {urdf_file_path}")
    print(f"Meshes path: {meshes_path}")
    
    # Replace the package:// path with the absolute meshes path
    urdf_content = urdf_content.replace('package://resource/robot/pm_v2/meshes/', 
                                       meshes_path + '/')
    
    # Create a temporary file with fixed paths
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    temp_file.write(urdf_content)
    temp_file.close()
    
    return temp_file.name

def urdf_to_mujoco_xml(urdf_path, original_urdf_path=None, contact_forces=None):
    """Convert URDF to Mujoco XML format with proper joint hierarchy and contact force spheres"""
    try:
        # Read URDF file directly
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        
        # Parse URDF to extract mesh information
        root = ET.fromstring(urdf_content)
        
        # Get mesh directory path - use original path if provided (for temp files)
        if original_urdf_path:
            base_path = os.path.dirname(original_urdf_path)  # urdf directory
        else:
            base_path = os.path.dirname(urdf_path)  # urdf directory
        robot_path = os.path.dirname(base_path)  # pm_v2 directory
        meshes_path = os.path.join(robot_path, 'meshes')
        meshes_path = os.path.abspath(meshes_path)
        
        # Use absolute path for Mujoco
        absolute_mesh_path = meshes_path
        
        print(f"Debug - urdf_path: {urdf_path}")
        print(f"Debug - original_urdf_path: {original_urdf_path}")
        print(f"Debug - base_path: {base_path}")
        print(f"Debug - robot_path: {robot_path}")
        print(f"Debug - meshes_path: {meshes_path}")
        print(f"Debug - absolute_mesh_path: {absolute_mesh_path}")
        
        # Verify the path exists
        if not os.path.exists(absolute_mesh_path):
            print(f"ERROR: Mesh directory does not exist: {absolute_mesh_path}")
            return None
        
        # List some mesh files to verify
        mesh_files = os.listdir(absolute_mesh_path)
        print(f"Debug - Found {len(mesh_files)} files in mesh directory")
        print(f"Debug - First 5 mesh files: {mesh_files[:5]}")
        
        # Create basic Mujoco XML with correct mesh directory
        xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="pm_v2">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.0001" iterations="50" solver="Newton" tolerance="1e-10"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="robot" rgba="0.8 0.8 0.8 1"/>
    <material name="contact_force" rgba="1 0 0 0.8"/>
'''
        
        # Add mesh definitions to asset section
        mesh_files = os.listdir(absolute_mesh_path)
        for mesh_file in mesh_files:
            if mesh_file.endswith('.STL'):
                mesh_name = mesh_file.replace('.STL', '')
                xml_content += f'    <mesh name="{mesh_name}" file="{absolute_mesh_path}/{mesh_file}"/>\n'
        
        xml_content += '''  </asset>
  
  <worldbody>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" castshadow="false" pos="0 0 5" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 0" size="0 0 .025" rgba=".9 .9 .9 1" material="grid"/>
    
    <!-- Robot body with proper joint hierarchy - matching XML coordinate system -->
    <body name="LINK_BASE" pos="0 0 0.82">
      <freejoint/>
'''
        
        # Extract links and joints from URDF
        links = {}
        joints = {}
        
        # First pass: collect all links
        for link in root.findall('.//link'):
            link_name = link.get('name')
            links[link_name] = link
        
        # Second pass: collect all joints
        for joint in root.findall('.//joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # Get joint origin
            origin = joint.find('origin')
            pos = "0 0 0"
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                pos = xyz
            
            # Get joint axis
            axis = joint.find('axis')
            axis_xyz = "0 0 1"
            if axis is not None:
                axis_xyz = axis.get('xyz', '0 0 1')
            
            joints[joint_name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'pos': pos,
                'axis': axis_xyz
            }
        
        # Build body hierarchy starting from base
        processed_links = set()
        mesh_count = 0
        
        def add_link_to_xml(link_name, indent=6):
            nonlocal xml_content, mesh_count
            if link_name in processed_links:
                return
            
            processed_links.add(link_name)
            link = links.get(link_name)
            if link is None:
                return
            
            # Find visual geometry
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename', '')
                        if mesh_filename:
                            # Extract mesh name from filename - remove .STL extension
                            mesh_name = os.path.basename(mesh_filename).replace('.STL', '')
                            print(f"Debug - Link: {link_name}, Mesh: {mesh_name}")
                            xml_content += f'{" " * indent}<geom name="{link_name}" type="mesh" mesh="{mesh_name}" material="robot"/>\n'
                            mesh_count += 1
            
            # Find child joints
            for joint_name, joint_info in joints.items():
                if joint_info['parent'] == link_name:
                    child_link = joint_info['child']
                    joint_type = joint_info['type']
                    joint_pos = joint_info['pos']
                    joint_axis = joint_info['axis']
                    
                    # Add joint and child body
                    xml_content += f'{" " * indent}<body name="{child_link}" pos="{joint_pos}">\n'
                    
                    if joint_type == 'revolute':
                        xml_content += f'{" " * (indent + 2)}<joint name="{joint_name}" axis="{joint_axis}"/>\n'
                    elif joint_type == 'prismatic':
                        xml_content += f'{" " * (indent + 2)}<joint name="{joint_name}" type="slide" axis="{joint_axis}"/>\n'
                    
                    # Recursively add child link
                    add_link_to_xml(child_link, indent + 2)
                    
                    xml_content += f'{" " * indent}</body>\n'
        
        # Start with base link
        add_link_to_xml('LINK_BASE')
        
        xml_content += '''    </body>
    
    <!-- Contact Force Spheres -->
'''
        
        # Add contact force spheres if available
        if contact_forces:
            max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
            print(f"Adding {len(contact_forces)} contact force spheres to XML...")
            
            for i, cf in enumerate(contact_forces):
                radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
                pos = cf['position']
                force = cf['max_force']
                
                xml_content += f'''    <body name="contact_sphere_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" material="contact_force"/>
      <site name="contact_site_{i}" pos="0 0 0" size="0.001"/>
    </body>
'''
        
        xml_content += '''  </worldbody>
  
  <keyframe/>
</mujoco>'''
        
        print(f"Created Mujoco XML with {mesh_count} mesh components")
        if contact_forces:
            print(f"Added {len(contact_forces)} contact force spheres")
        print(f"Mesh directory: {meshes_path}")
        
        # Debug: Print first few lines of XML
        xml_lines = xml_content.split('\n')
        print("Generated XML (first 10 lines):")
        for i, line in enumerate(xml_lines[:10]):
            print(f"  {i+1}: {line}")
        
        return xml_content
        
    except Exception as e:
        print(f"Error converting URDF to Mujoco XML: {e}")
        return None

def create_contact_force_visualization(df, x_col, y_col, z_col):
    """Create contact force visualization data with enhanced analysis"""
    print("Processing contact force data...")
    
    # Filter valid coordinates
    valid_mask = (df[x_col].notna() & df[y_col].notna() & df[z_col].notna() & 
                  np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[z_col]))
    
    if not valid_mask.any():
        print("No valid contact coordinates found")
        return None
    
    df_valid = df[valid_mask].copy()
    
    # Calculate force magnitude
    if 'force_magnitude' not in df_valid.columns:
        df_valid['force_magnitude'] = np.sqrt(df_valid['force_x']**2 + df_valid['force_y']**2 + df_valid['force_z']**2)
    
    # Coordinate system adjustment
    # XML model has robot base at z=0.82, but contact points are relative to robot base
    # We need to understand if contact points are in robot base coordinates or world coordinates
    print(f"Original coordinate ranges:")
    print(f"  X: {df_valid[x_col].min():.3f} to {df_valid[x_col].max():.3f}")
    print(f"  Y: {df_valid[y_col].min():.3f} to {df_valid[y_col].max():.3f}")
    print(f"  Z: {df_valid[z_col].min():.3f} to {df_valid[z_col].max():.3f}")
    
    # Check if contact points are already in robot base coordinates
    # If Z coordinates are mostly negative or close to 0, they might be relative to ground
    # If Z coordinates are positive and reasonable for robot height, they might be relative to robot base
    z_mean = df_valid[z_col].mean()
    z_std = df_valid[z_col].std()
    print(f"Z coordinate statistics: mean={z_mean:.3f}, std={z_std:.3f}")
    
    # If Z coordinates are mostly negative or very small, they are relative to ground
    # We need to convert them to robot base coordinates
    # XML model has robot base at z=0.82, so we add 0.82 to ground coordinates
    if z_mean < 0.1:  # Most contact points are near ground level
        print("Contact points appear to be relative to ground. Converting to robot base coordinates...")
        # Convert from ground coordinates to robot base coordinates
        # Robot base is at z=0.82 in world coordinates (matching XML)
        df_valid['z_adjusted'] = df_valid[z_col] + 0.82
        print(f"Adjusted Z coordinate range: {df_valid['z_adjusted'].min():.3f} to {df_valid['z_adjusted'].max():.3f}")
    else:
        print("Contact points appear to be already in robot base coordinates.")
        df_valid['z_adjusted'] = df_valid[z_col]
    
    # Handle potential coordinate axis misalignment
    # The contact data might have different axis definitions than the URDF model
    # Try different coordinate transformations to align them properly
    
    # Option 1: Swap X and Y axes (common in robotics where X is forward, Y is left)
    print("Attempting coordinate axis alignment...")
    
    # Check the range of X and Y to determine if we need to swap axes
    x_range = df_valid[x_col].max() - df_valid[x_col].min()
    y_range = df_valid[y_col].max() - df_valid[y_col].min()
    
    print(f"X range: {x_range:.3f}, Y range: {y_range:.3f}")
    
    # If X range is much larger than Y range, it might be the forward direction
    # If Y range is much larger than X range, it might be the forward direction
    # Let's try swapping X and Y to see if it aligns better
    if x_range > y_range * 1.5:
        print("X range is significantly larger than Y range. X might be the forward direction.")
        # Keep current axis assignment
        df_valid['x_adjusted'] = df_valid[x_col]
        df_valid['y_adjusted'] = df_valid[y_col]
    elif y_range > x_range * 1.5:
        print("Y range is significantly larger than X range. Y might be the forward direction.")
        # Swap X and Y
        df_valid['x_adjusted'] = df_valid[y_col]
        df_valid['y_adjusted'] = df_valid[x_col]
    else:
        print("X and Y ranges are similar. Using original coordinate system.")
        # Keep original coordinate system
        df_valid['x_adjusted'] = df_valid[x_col]
        df_valid['y_adjusted'] = df_valid[y_col]
    
    print(f"Adjusted coordinate ranges:")
    print(f"  X: {df_valid['x_adjusted'].min():.3f} to {df_valid['x_adjusted'].max():.3f}")
    print(f"  Y: {df_valid['y_adjusted'].min():.3f} to {df_valid['y_adjusted'].max():.3f}")
    print(f"  Z: {df_valid['z_adjusted'].min():.3f} to {df_valid['z_adjusted'].max():.3f}")
    
    # Get unique contact positions with adjusted coordinates
    contact_positions = df_valid[['x_adjusted', 'y_adjusted', 'z_adjusted']].values
    unique_positions = np.unique(contact_positions, axis=0)
    
    # Enhanced contact point analysis
    print(f"Found {len(unique_positions)} unique contact positions")
    print(f"Force range: {df_valid['force_magnitude'].min():.2f} - {df_valid['force_magnitude'].max():.2f} N")
    print(f"Average force: {df_valid['force_magnitude'].mean():.2f} N")
    
    # Limit number of contact points for performance but increase from 50 to 200
    if len(unique_positions) > 200:
        print(f"Too many contact points ({len(unique_positions)}), sampling 200 for better visualization...")
        indices = np.linspace(0, len(unique_positions)-1, 200, dtype=int)
        unique_positions = unique_positions[indices]
    
    contact_forces = []
    for pos in unique_positions:
        # Find matching positions in original data
        mask = (df_valid['x_adjusted'] == pos[0]) & (df_valid['y_adjusted'] == pos[1]) & (df_valid['z_adjusted'] == pos[2])
        forces_at_pos = df_valid.loc[mask, 'force_magnitude']
        max_force_at_pos = forces_at_pos.max()
        avg_force_at_pos = forces_at_pos.mean()
        contact_forces.append({
            'position': pos,  # Use adjusted position
            'max_force': max_force_at_pos,
            'avg_force': avg_force_at_pos,
            'contact_count': len(forces_at_pos)
        })
    
    print(f"Created {len(contact_forces)} contact force visualizations")
    return contact_forces

def render_with_contacts(model, data, contact_forces):
    """Render model with contact forces using official Mujoco renderer"""
    print("Rendering with contact forces...")
    
    # Create renderer
    renderer = mj.Renderer(model)
    
    # Set up scene options
    scene_option = mj.MjvOption()
    scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    
    # Render the scene
    renderer.update_scene(data, scene_option=scene_option)
    pixels = renderer.render()
    
    # Save the rendered image
    save_path = "logs/urdf_contact_forces_mujoco.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_image(save_path, pixels)
    print(f"Rendered image saved to: {save_path}")
    
    # Also save contact force data for visualization
    if contact_forces:
        contact_data = []
        max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
        
        for cf in contact_forces:
            radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
            contact_data.append({
                'position': cf['position'].tolist(),
                'radius': radius,
                'force': cf['max_force']
            })
        
        # Save contact data as JSON for external visualization
        import json
        contact_save_path = "logs/contact_forces_data.json"
        with open(contact_save_path, 'w') as f:
            json.dump(contact_data, f, indent=2)
        print(f"Contact force data saved to: {contact_save_path}")
    
    return pixels

def show_mujoco_viewer(model, data, contact_forces):
    """Show Mujoco viewer with interactive interface and contact force spheres"""
    print("Opening Mujoco viewer...")
    
    # Import viewer module
    try:
        from mujoco import viewer
    except ImportError:
        print("Error: mujoco.viewer module not available")
        print("Please install mujoco with viewer support")
        return None
    
    # Save contact force data for external analysis
    if contact_forces:
        max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
        print(f"Found {len(contact_forces)} contact points:")
        
        # Save contact force data as JSON
        import json
        contact_data = []
        for i, cf in enumerate(contact_forces):
            radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
            pos = cf['position']
            
            contact_data.append({
                'id': i,
                'position': pos.tolist(),
                'max_force': cf['max_force'],
                'avg_force': cf.get('avg_force', cf['max_force']),
                'contact_count': cf.get('contact_count', 1),
                'radius': radius
            })
            
            print(f"  Contact {i+1}: position={pos}, force={cf['max_force']:.2f}, radius={radius:.4f}")
        
        # Save to JSON file
        json_path = "logs/contact_forces_visualization.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({
                'contact_points': contact_data,
                'max_force': max_force,
                'total_contacts': len(contact_forces),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Contact force data saved to: {json_path}")
    
    print("Launching Mujoco viewer...")
    print("Controls:")
    print("- Rotate: Left mouse button")
    print("- Pan: Right mouse button") 
    print("- Zoom: Mouse wheel")
    print("- Close: Press 'ESC' or close the window")
    print("\nContact Force Visualization:")
    print("- Red spheres represent contact points")
    print("- Sphere size proportional to force magnitude")
    print("- Larger spheres = higher contact forces")
    print("\nNote: Mujoco built-in contact visualization uses cylinders (not spheres)")
    print("      This is the standard way to visualize contact points in Mujoco")
    
    # Launch the viewer with contact force visualization enabled
    try:
        # Launch viewer first
        handle = viewer.launch_passive(model, data)
        print("Viewer launched successfully!")
        
        # Set up visualization options after launch
        if hasattr(handle, 'opt'):
            # Enable contact point visualization
            # mjVIS_CONTACTPOINT = 8, mjVIS_CONTACTFORCE = 10
            handle.opt.flags[8] = True   # mjVIS_CONTACTPOINT
            handle.opt.flags[10] = True  # mjVIS_CONTACTFORCE
            print("Contact force visualization enabled!")
            print("Note: Contact points are displayed as cylinders (Mujoco standard)")
        else:
            print("Warning: Could not access visualization options")
        
        # Keep the viewer open until user closes it
        while handle.is_running():
            time.sleep(0.1)
        
        print("Viewer closed")
        return handle
        
    except Exception as e:
        print(f"Error launching viewer: {e}")
        return None

def show_mujoco_viewer_with_spheres(model, data, contact_forces):
    """Show Mujoco viewer with interactive interface and contact force spheres"""
    print("Opening Mujoco viewer...")
    
    # Import viewer module
    try:
        from mujoco import viewer
    except ImportError:
        print("Error: mujoco.viewer module not available")
        print("Please install mujoco with viewer support")
        return None
    
    # Save contact force data for external analysis
    if contact_forces:
        max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
        print(f"Found {len(contact_forces)} contact points:")
        
        # Save contact force data as JSON
        import json
        contact_data = []
        for i, cf in enumerate(contact_forces):
            radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
            pos = cf['position']
            
            contact_data.append({
                'id': i,
                'position': pos.tolist(),
                'max_force': cf['max_force'],
                'avg_force': cf.get('avg_force', cf['max_force']),
                'contact_count': cf.get('contact_count', 1),
                'radius': radius
            })
            
            print(f"  Contact {i+1}: position={pos}, force={cf['max_force']:.2f}, radius={radius:.4f}")
        
        # Save to JSON file
        json_path = "logs/contact_forces_visualization.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({
                'contact_points': contact_data,
                'max_force': max_force,
                'total_contacts': len(contact_forces),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Contact force data saved to: {json_path}")
    
    print("Launching Mujoco viewer...")
    print("Controls:")
    print("- Rotate: Left mouse button")
    print("- Pan: Right mouse button") 
    print("- Zoom: Mouse wheel")
    print("- Close: Press 'ESC' or close the window")
    print("\nContact Force Visualization:")
    print("- Red spheres represent contact points")
    print("- Sphere size proportional to force magnitude")
    print("- Larger spheres = higher contact forces")
    
    # Launch the viewer
    try:
        handle = viewer.launch_passive(model, data)
        print("Viewer launched successfully!")
        
        # Note: Contact force spheres are not visible in the viewer
        # because we cannot dynamically add them to the scene
        # The spheres are saved to JSON for external visualization
        if contact_forces:
            print("Note: Contact force spheres are saved to JSON but not visible in viewer")
            print("Use external visualization tools to view the contact force data")
        
        # Keep the viewer open until user closes it
        while handle.is_running():
            time.sleep(0.1)
        
        print("Viewer closed")
        return handle
        
    except Exception as e:
        print(f"Error launching viewer: {e}")
        return None

def show_mujoco_viewer_with_sphere_contacts(model, data, contact_forces):
    """Show Mujoco viewer with sphere-shaped contact force visualization"""
    print("Opening Mujoco viewer with sphere contact visualization...")
    
    # Import viewer module
    try:
        from mujoco import viewer
    except ImportError:
        print("Error: mujoco.viewer module not available")
        print("Please install mujoco with viewer support")
        return None
    
    # Save contact force data for external analysis
    if contact_forces:
        max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
        print(f"Found {len(contact_forces)} contact points:")
        
        # Save contact force data as JSON
        import json
        contact_data = []
        for i, cf in enumerate(contact_forces):
            radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
            pos = cf['position']
            
            contact_data.append({
                'id': i,
                'position': pos.tolist(),
                'max_force': cf['max_force'],
                'avg_force': cf.get('avg_force', cf['max_force']),
                'contact_count': cf.get('contact_count', 1),
                'radius': radius
            })
            
            print(f"  Contact {i+1}: position={pos}, force={cf['max_force']:.2f}, radius={radius:.4f}")
        
        # Save to JSON file
        json_path = "logs/contact_forces_visualization.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({
                'contact_points': contact_data,
                'max_force': max_force,
                'total_contacts': len(contact_forces),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Contact force data saved to: {json_path}")
    
    print("Launching Mujoco viewer...")
    print("Controls:")
    print("- Rotate: Left mouse button")
    print("- Pan: Right mouse button") 
    print("- Zoom: Mouse wheel")
    print("- Close: Press 'ESC' or close the window")
    print("\nContact Force Visualization:")
    print("- Red spheres represent contact points")
    print("- Sphere size proportional to force magnitude")
    print("- Larger spheres = higher contact forces")
    print("\nNote: Using sphere-shaped contact visualization (custom implementation)")
    print("      Use URDF input for sphere visualization, or modify XML manually")
    
    # Launch the viewer
    try:
        handle = viewer.launch_passive(model, data)
        print("Viewer launched successfully!")
        print("Note: Sphere contact visualization requires XML modification")
        print("      Use URDF input for sphere visualization, or modify XML manually")
        
        # Keep the viewer open until user closes it
        while handle.is_running():
            time.sleep(0.1)
        
        print("Viewer closed")
        return handle
        
    except Exception as e:
        print(f"Error launching viewer: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python3 mujoco_urdf_contact_display.py <csv_file> <xml_or_urdf_file> [sphere|cylinder] [world|urdf]")
        print("Example: python3 mujoco_urdf_contact_display.py logs/contact_data_20250723_155151.csv src/simulation/mujoco/assets/resource/robot/pm_v2/xml/serial_pm_v2.xml")
        print("Example: python3 mujoco_urdf_contact_display.py logs/contact_data_20250723_155151.csv src/simulation/mujoco/assets/resource/pm_v2.xml sphere")
        print("Example: python3 mujoco_urdf_contact_display.py logs/contact_data_20250723_155151.csv src/simulation/mujoco/assets/resource/robot/pm_v2/urdf/serial_pm_v2.urdf sphere world")
        print("Note: The script will automatically detect if you provide an XML file and use it directly")
        print("      If you provide a URDF file, it will convert it to XML first")
        print("      Optional third argument: 'sphere' for sphere visualization, 'cylinder' for Mujoco built-in (default)")
        print("      Optional fourth argument: 'world' for world coordinates (pos_x/y/z), 'urdf' for URDF coordinates (urdf_x/y/z_body1) (default)")
        return
    
    csv_file = sys.argv[1]
    model_file = sys.argv[2]
    
    # Check for visualization type
    viz_type = "cylinder"  # default to Mujoco built-in
    coord_type = "urdf"    # default to URDF coordinates
    if len(sys.argv) >= 4:
        viz_type = sys.argv[3].lower()
        if viz_type not in ["sphere", "cylinder"]:
            print("Error: Visualization type must be 'sphere' or 'cylinder'")
            return
    
    if len(sys.argv) >= 5:
        coord_type = sys.argv[4].lower()
        if coord_type not in ["world", "urdf"]:
            print("Error: Coordinate type must be 'world' or 'urdf'")
            return
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return
    
    # Load contact data
    print(f"Loading contact data: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
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
        # Use URDF coordinates
        urdf_columns = [col for col in df.columns if 'urdf' in col.lower()]
        if not urdf_columns:
            print("No URDF coordinate columns found")
            return
        
        # Use corrected body1 coordinates if available (new format)
        if 'urdf_corrected_x_body1' in df.columns and 'urdf_corrected_y_body1' in df.columns and 'urdf_corrected_z_body1' in df.columns:
            x_col, y_col, z_col = 'urdf_corrected_x_body1', 'urdf_corrected_y_body1', 'urdf_corrected_z_body1'
            print("Using corrected URDF coordinates (urdf_corrected_x/y/z_body1)")
        # Fallback to old format if new format not available
        elif 'urdf_x_body1' in df.columns and 'urdf_y_body1' in df.columns and 'urdf_z_body1' in df.columns:
            x_col, y_col, z_col = 'urdf_x_body1', 'urdf_y_body1', 'urdf_z_body1'
            print("Using legacy URDF coordinates (urdf_x/y/z_body1)")
        else:
            print("No complete URDF coordinate sets found")
            return
    
    # Create contact force visualization
    contact_forces = create_contact_force_visualization(df, x_col, y_col, z_col)
    
    # Determine if model file is XML or URDF
    if model_file.endswith('.xml'):
        # Direct XML file
        xml_path = model_file
        print(f"Using existing XML file: {xml_path}")
        
        # Get the directory of the XML file to handle relative paths
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        original_dir = os.getcwd()
        
        try:
            # Change to XML directory to handle relative includes
            os.chdir(xml_dir)
            print(f"Changed to directory: {xml_dir}")
            
            # Check if XML contains includes (which would make modification complex)
            with open(os.path.basename(xml_path), 'r') as f:
                xml_content = f.read()
            
            has_includes = '<include' in xml_content
            
            if has_includes and contact_forces:
                print("XML contains includes, using original file and adding spheres in viewer...")
                # Load original XML without modification
                model = mj.MjModel.from_xml_path(os.path.basename(xml_path))
                data = mj.MjData(model)
                
                # Show interactive viewer with contact forces
                if viz_type == "sphere":
                    print("Warning: Sphere visualization not available for XML with includes")
                    print("         Using cylinder visualization instead")
                    show_mujoco_viewer(model, data, contact_forces)
                else:
                    show_mujoco_viewer(model, data, contact_forces)
            else:
                # Add contact force spheres to XML if available
                if contact_forces:
                    print(f"Adding {len(contact_forces)} contact force spheres to XML...")
                    
                    # Find the end of worldbody section
                    worldbody_end = xml_content.find('</worldbody>')
                    if worldbody_end == -1:
                        print("Warning: Could not find </worldbody> tag, adding spheres at end")
                        worldbody_end = xml_content.find('</mujoco>')
                        if worldbody_end == -1:
                            print("Error: Could not find </mujoco> tag")
                            return
                    
                    # Prepare contact spheres XML
                    spheres_xml = '\n    <!-- Contact Force Spheres -->\n'
                    max_force = max(cf['max_force'] for cf in contact_forces) if contact_forces else 1.0
                    
                    for i, cf in enumerate(contact_forces):
                        radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
                        pos = cf['position']
                        
                        spheres_xml += f'''    <body name="contact_sphere_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" rgba="1.0 0.0 0.0 0.8"/>
      <site name="contact_site_{i}" pos="0 0 0" size="0.001"/>
    </body>
'''
                    
                    # Insert spheres before </worldbody>
                    xml_content = xml_content[:worldbody_end] + spheres_xml + xml_content[worldbody_end:]
                    
                    # Create temporary XML file with contact spheres
                    xml_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
                    xml_temp.write(xml_content)
                    xml_temp.close()
                    
                    # Load Mujoco model from modified XML
                    print("Loading Mujoco model from modified XML...")
                    model = mj.MjModel.from_xml_path(xml_temp.name)
                    data = mj.MjData(model)
                    
                    # Clean up temporary file
                    os.unlink(xml_temp.name)
                else:
                    # Load Mujoco model directly
                    print("Loading Mujoco model from XML...")
                    model = mj.MjModel.from_xml_path(os.path.basename(xml_path))
                    data = mj.MjData(model)
                
                # Show interactive viewer
                if viz_type == "sphere":
                    show_mujoco_viewer_with_sphere_contacts(model, data, contact_forces)
                else:
                    show_mujoco_viewer(model, data, contact_forces)
            
        except Exception as e:
            print(f"Error loading XML: {e}")
        finally:
            # Restore original directory
            os.chdir(original_dir)
        
    elif model_file.endswith('.urdf'):
        # URDF file - need to convert
        urdf_path = model_file
        print(f"Converting URDF to XML: {urdf_path}")
        
        # Fix URDF paths
        fixed_urdf_path = fix_urdf_paths(urdf_path)
        
        # Convert to Mujoco XML with contact forces
        xml_content = urdf_to_mujoco_xml(fixed_urdf_path, urdf_path, contact_forces)
        
        # Clean up temporary URDF file
        os.unlink(fixed_urdf_path)
        
        if xml_content is None:
            print("Failed to convert URDF to Mujoco XML")
            return
        
        # Create temporary XML file
        xml_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        xml_temp.write(xml_content)
        xml_temp.close()
        
        try:
            # Load Mujoco model
            print("Loading Mujoco model from converted XML...")
            model = mj.MjModel.from_xml_path(xml_temp.name)
            data = mj.MjData(model)
            
            # Show interactive viewer
            if viz_type == "sphere":
                show_mujoco_viewer_with_sphere_contacts(model, data, contact_forces)
            else:
                show_mujoco_viewer(model, data, contact_forces)
            
        except Exception as e:
            print(f"Error in Mujoco display: {e}")
        finally:
            # Clean up temporary XML file
            os.unlink(xml_temp.name)
    else:
        print("Model file must be either .xml or .urdf")
        return
    
    print("Viewer closed successfully!")

if __name__ == "__main__":
    main() 