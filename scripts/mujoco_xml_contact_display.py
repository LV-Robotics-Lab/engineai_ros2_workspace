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
import mujoco as mj
import tempfile
import xml.etree.ElementTree as ET
import mediapy as media
from datetime import datetime

def analyze_csv_data(df):
    """Analyze CSV data and print statistics"""
    print("\n=== CSV Data Analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for key columns
    key_columns = {
        'Time': ['sim_time'],
        'Contact Info': ['contact_id', 'geom1_name', 'geom2_name'],
        'World Position': ['pos_x', 'pos_y', 'pos_z'],
        'Robot Frame Position': ['robot_frame_x', 'robot_frame_y', 'robot_frame_z'],
        'Forces': ['force_x', 'force_y', 'force_z', 'force_magnitude'],
        'Torques': ['torque_x', 'torque_y', 'torque_z'],
        'Base Link': ['base_link_x', 'base_link_y', 'base_link_z', 'base_link_quat_w', 'base_link_quat_x', 'base_link_quat_y', 'base_link_quat_z']
    }
    
    for category, columns in key_columns.items():
        found_columns = [col for col in columns if col in df.columns]
        if found_columns:
            print(f"{category}: {found_columns}")
            if category == 'Time' and 'sim_time' in df.columns:
                print(f"  Time range: {df['sim_time'].min():.3f} - {df['sim_time'].max():.3f} seconds")
            elif category == 'Forces' and 'force_magnitude' in df.columns:
                print(f"  Force range: {df['force_magnitude'].min():.3f} - {df['force_magnitude'].max():.3f} N")
        else:
            print(f"{category}: Not found")
    
    print("=" * 30)

def create_contact_force_visualization(df, x_col, y_col, z_col):
    """Create contact force visualization data with sphere sizes based on force magnitude"""
    print("Processing contact force data for sphere visualization...")
    
    # Filter valid coordinates
    valid_mask = (df[x_col].notna() & df[y_col].notna() & df[z_col].notna() & 
                  np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[z_col]))
    
    if not valid_mask.any():
        print("No valid contact coordinates found")
        return []
    
    df_valid = df[valid_mask].copy()
    print(f"Found {len(df_valid)} valid contact points")
    
    # Group by position to aggregate forces at the same location
    position_groups = {}
    
    for _, row in df_valid.iterrows():
        pos = (row[x_col], row[y_col], row[z_col])
        
        # Use force magnitude if available, otherwise calculate from components
        if 'force_magnitude' in row:
            force_mag = row['force_magnitude']
        elif all(col in row for col in ['force_x', 'force_y', 'force_z']):
            force_mag = np.sqrt(row['force_x']**2 + row['force_y']**2 + row['force_z']**2)
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
    
    # Sort by force magnitude (highest first) and limit the number of spheres
    contact_forces.sort(key=lambda x: x['max_force'], reverse=True)
    
    # Limit to maximum 1000 spheres to avoid memory issues
    max_spheres = 1000
    if len(contact_forces) > max_spheres:
        print(f"Too many contact points ({len(contact_forces)}), limiting to {max_spheres} highest force points")
        contact_forces = contact_forces[:max_spheres]
    
    print(f"Created {len(contact_forces)} unique contact points for visualization")
    if contact_forces:
        max_force = max(cf['max_force'] for cf in contact_forces)
        min_force = min(cf['max_force'] for cf in contact_forces)
        print(f"Force range: {min_force:.3f} - {max_force:.3f} N")
    
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
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"/>
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
        
        # Add contact force information to viewer
        if contact_forces:
            max_force = max(cf['max_force'] for cf in contact_forces)
            print(f"\nContact Force Visualization:")
            print(f"Total contact points: {len(contact_forces)}")
            print(f"Max force: {max_force:.3f} N")
            print(f"Sphere sizes represent force magnitude")
            print(f"Colors: Red (high force) to Yellow (low force)")
        
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

def load_mujoco_model_with_contact_spheres(xml_file, contact_forces):
    """Load Mujoco model with contact spheres, handling relative paths correctly"""
    print(f"Loading Mujoco model from: {xml_file}")
    
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
            # Calculate sphere radius based on force magnitude
            radius = 0.01 + 0.04 * (cf['max_force'] / max_force) if max_force > 0 else 0.01
            pos = cf['position']
            
            # Color based on force magnitude (red to yellow)
            force_ratio = cf['max_force'] / max_force if max_force > 0 else 0
            r = 1.0
            g = force_ratio
            b = 0.0
            a = 0.8
            
            spheres_xml += f'''    <body name="contact_sphere_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">
      <geom name="contact_geom_{i}" type="sphere" size="{radius:.6f}" rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"/>
      <site name="contact_site_{i}" pos="0 0 0" size="0.001"/>
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
        print("Usage: python3 mujoco_xml_contact_display.py <csv_file> <xml_file> [world|robot_frame]")
        print("")
        print("Examples:")
        print("  # Using robot frame coordinates (default)")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml")
        print("")
        print("  # Using world coordinates")
        print("  python3 mujoco_xml_contact_display.py logs/contact_data.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml world")
        print("")
        print("CSV Format Support:")
        print("  - New format: sim_time, contact_id, geom1_name, geom2_name, pos_x/y/z, robot_frame_x/y/z, force_magnitude, etc.")
        print("  - Legacy format: pos_x/y/z, force_x/y/z, etc.")
        print("")
        print("Arguments:")
        print("  csv_file: Path to the CSV file with contact force data")
        print("  xml_file: Path to XML model file (e.g., pm_v2_mesh.xml)")
        print("  world|robot_frame: Coordinate system (default: robot_frame)")
        print("")
        print("Note: Contact forces will be displayed as spheres with sizes proportional to force magnitude")
        return
    
    csv_file = sys.argv[1]
    xml_file = sys.argv[2]
    
    # Check for coordinate type
    coord_type = "robot_frame"    # default to robot frame coordinates
    if len(sys.argv) >= 4:
        coord_type = sys.argv[3].lower()
        if coord_type not in ["world", "robot_frame"]:
            print("Error: Coordinate type must be 'world' or 'robot_frame'")
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
        # Use robot frame coordinates
        if 'robot_frame_x' in df.columns and 'robot_frame_y' in df.columns and 'robot_frame_z' in df.columns:
            x_col, y_col, z_col = 'robot_frame_x', 'robot_frame_y', 'robot_frame_z'
            print("Using robot frame coordinates (robot_frame_x/y/z)")
        else:
            print("Robot frame coordinates (robot_frame_x/y/z) not found in CSV")
            return
    
    # Create contact force visualization
    contact_forces = create_contact_force_visualization(df, x_col, y_col, z_col)
    
    if not contact_forces:
        print("No contact forces to display")
        return
    
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