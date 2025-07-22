import laspy
import numpy as np
import open3d as o3d
import os

# Load the LAZ file
laz_file_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"
las = laspy.read(laz_file_path)

# Extract X, Y, Z coordinates
xyz = np.vstack((las.x, las.y, las.z)).T

# Print the total number of points and ranges for debugging
print(f"Total number of points: {len(xyz)}")
print(f"X range (min, max): ({np.min(xyz[:, 0])}, {np.max(xyz[:, 0])})")
print(f"Y range (min, max): ({np.min(xyz[:, 1])}, {np.max(xyz[:, 1])})")
print(f"Z range (min, max): ({np.min(xyz[:, 2])}, {np.max(xyz[:, 2])})")

# Remove the adjacent road (adjust z_threshold based on data)
z_threshold = 90  # Adjusted to 90, verify with your Z range
road_mask = xyz[:, 2] <= z_threshold
xyz_road_removed = xyz[~road_mask]
print(f"Points after road removal: {len(xyz_road_removed)}")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_road_removed)

# Downsample for efficiency (adjust voxel_size as needed)
pcd = pcd.voxel_down_sample(voxel_size=0.1)

# Manual selection of points to define the region to remove
print("Please select points on the river/strip to remove using Shift + Left Click. Press 'Q' to finish.")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Select River/Strip Points to Remove")
vis.add_geometry(pcd)
vis.run()  # Interactive selection
vis.destroy_window()

# Get selected point indices
selected_indices = vis.get_picked_points()
if len(selected_indices) < 3:
    print("Not enough points selected. Please select at least 3 points and rerun.")
else:
    selected_points = np.asarray(pcd.points)[selected_indices]

    # Define a bounding box or convex hull from selected points (simplified bounding box)
    x_min = np.min(selected_points[:, 0])
    x_max = np.max(selected_points[:, 0])
    y_min = np.min(selected_points[:, 1])
    y_max = np.max(selected_points[:, 1])
    z_min = np.min(selected_points[:, 2])
    z_max = np.max(selected_points[:, 2])

    print(f"Masking region: X=({x_min}, {x_max}), Y=({y_min}, {y_max}), Z=({z_min}, {z_max})")

    # Create a mask to remove the selected region
    removal_mask = (xyz_road_removed[:, 0] >= x_min) & (xyz_road_removed[:, 0] <= x_max) & \
                   (xyz_road_removed[:, 1] >= y_min) & (xyz_road_removed[:, 1] <= y_max) & \
                   (xyz_road_removed[:, 2] >= z_min) & (xyz_road_removed[:, 2] <= z_max)
    xyz_cleaned = xyz_road_removed[~removal_mask]

    # Print points before and after removal
    print(f"Points before river/strip removal: {len(xyz_road_removed)}")
    print(f"Points after river/strip removal: {len(xyz_cleaned)}")

    # Create new point cloud for visualization
    pcd_cleaned = o3d.geometry.PointCloud()
    pcd_cleaned.points = o3d.utility.Vector3dVector(xyz_cleaned)

    # Visualize the result
    print("Visualizing: Green = Remaining Embankment")
    pcd_cleaned.paint_uniform_color([0, 1, 0])  # Green for remaining embankment
    o3d.visualization.draw_geometries([pcd_cleaned], window_name="Embankment with Selected Region Removed")

    # Save the cleaned point cloud
    output_path = "C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06_cleaned.laz"
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create LAS data with proper header and points
        las_cleaned = laspy.LasData(las.header)
        # Create points with the cleaned XYZ data, preserving the original point format
        points = np.zeros(len(xyz_cleaned), dtype=las.header.point_format.dtype)
        points[las.header.point_format.x] = xyz_cleaned[:, 0]
        points[las.header.point_format.y] = xyz_cleaned[:, 1]
        points[las.header.point_format.z] = xyz_cleaned[:, 2]
        las_cleaned.points = points
        las_cleaned.header.point_count = len(xyz_cleaned)

        # Write the cleaned LAS file
        las_cleaned.write(output_path)
        print(f"Cleaned point cloud saved as {output_path}")
        print(f"File exists: {os.path.exists(output_path)}")  # Debug check
    except AttributeError as e:
        print(f"AttributeError saving .laz file: {e}. This may indicate a laspy version issue. Updating laspy is recommended: pip install --upgrade laspy")
        # Fallback: Save as .xyz file
        xyz_output_path = output_path.replace('.laz', '_fallback.xyz')
        np.savetxt(xyz_output_path, xyz_cleaned, delimiter=' ', header='X Y Z', comments='')
        print(f"Failed to save .laz. Saved fallback as {xyz_output_path}")
    except Exception as e:
        print(f"Unexpected error saving .laz file: {e}")
        # Fallback: Save as .xyz file
        xyz_output_path = output_path.replace('.laz', '_fallback.xyz')
        np.savetxt(xyz_output_path, xyz_cleaned, delimiter=' ', header='X Y Z', comments='')
        print(f"Failed to save .laz. Saved fallback as {xyz_output_path}")