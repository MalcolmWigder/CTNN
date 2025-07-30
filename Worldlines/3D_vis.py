import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import os
import argparse
from scipy import ndimage
from skimage import measure
import plotly.graph_objects as go
import plotly.express as px

class Worldline3DVolumeVisualizer:
    def __init__(self, worldline_file="worldline_tensor.pt", npy_dir="./unpacked_npys"):
        """Load worldline data and mask files"""
        print(f"Loading worldline data from {worldline_file}...")
        
        self.data = torch.load(worldline_file)
        self.worldlines = self.data['worldlines']
        self.existence_mask = self.data['existence_mask']
        self.timestamps = self.data.get('timestamps', None)
        
        # Load mask metadata
        self.npy_dir = npy_dir
        self.mask_index_file = os.path.join(npy_dir, "mask_index.csv")
        if os.path.exists(self.mask_index_file):
            self.mask_meta = pd.read_csv(self.mask_index_file)
            print(f"Loaded mask metadata for {len(self.mask_meta)} frames")
        else:
            print("Warning: No mask metadata found")
            self.mask_meta = None
        
        print(f"Loaded {self.worldlines.shape[0]} worldlines")
        
    def load_mask(self, frame_idx):
        """Load the mask for a specific frame"""
        if self.mask_meta is None or frame_idx >= len(self.mask_meta):
            return None
            
        try:
            npy_file = self.mask_meta.iloc[frame_idx]['npy_file']
            mask_path = os.path.join(self.npy_dir, npy_file)
            mask = np.load(mask_path)
            return mask
        except Exception as e:
            print(f"Error loading mask for frame {frame_idx}: {e}")
            return None
    
    def extract_instance_mask(self, full_mask, centroid_x, centroid_y, tolerance=50):
        """Extract the instance mask closest to the given centroid"""
        if full_mask is None:
            return None
            
        unique_ids = np.unique(full_mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        if len(unique_ids) == 0:
            return None
        
        best_instance = None
        best_distance = float('inf')
        
        for instance_id in unique_ids:
            instance_mask = (full_mask == instance_id).astype(np.uint8)
            y_coords, x_coords = np.where(instance_mask)
            
            if len(x_coords) == 0:
                continue
                
            inst_centroid_x = np.mean(x_coords)
            inst_centroid_y = np.mean(y_coords)
            
            distance = np.sqrt((inst_centroid_x - centroid_x)**2 + 
                             (inst_centroid_y - centroid_y)**2)
            
            if distance < best_distance and distance < tolerance:
                best_distance = distance
                best_instance = instance_mask
        
        return best_instance
    
    def get_worldline_info(self, worldline_idx):
        """Get worldline information"""
        if worldline_idx >= len(self.worldlines):
            return None
            
        wl_mask = self.existence_mask[worldline_idx]
        active_frames = torch.where(wl_mask)[0]
        
        if len(active_frames) == 0:
            return None
            
        trajectory = self.worldlines[worldline_idx, active_frames]
        
        return {
            'index': worldline_idx,
            'active_frames': active_frames.numpy(),
            'centroid_x': trajectory[:, 0].numpy(),
            'centroid_y': trajectory[:, 1].numpy(),
            'area': trajectory[:, 2].numpy(),
        }
    
    def build_3d_volume(self, worldline_idx, downsample=2):
        """Build 3D volume with actual shapes over time"""
        info = self.get_worldline_info(worldline_idx)
        if info is None:
            return None
            
        print(f"Building 3D volume for worldline {worldline_idx}...")
        print(f"Loading {len(info['active_frames'])} frames...")
        
        # Collect all shape data
        shapes = []
        all_x_coords = []
        all_y_coords = []
        
        for i, frame_idx in enumerate(info['active_frames']):
            full_mask = self.load_mask(frame_idx)
            if full_mask is None:
                continue
                
            instance_mask = self.extract_instance_mask(
                full_mask, info['centroid_x'][i], info['centroid_y'][i]
            )
            
            if instance_mask is not None:
                # Downsample for performance
                if downsample > 1:
                    instance_mask = instance_mask[::downsample, ::downsample]
                
                y_coords, x_coords = np.where(instance_mask)
                
                if len(x_coords) > 0:
                    shapes.append({
                        'time_idx': i,
                        'frame_idx': frame_idx,
                        'x_coords': x_coords,
                        'y_coords': y_coords,
                        'mask': instance_mask
                    })
                    
                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
        
        if not shapes:
            print("No shapes found!")
            return None
        
        print(f"Found {len(shapes)} valid shapes")
        
        # Determine 3D volume dimensions
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        t_min, t_max = 0, len(shapes) - 1
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = x_max + padding
        y_min = max(0, y_min - padding)
        y_max = y_max + padding
        
        # Create 3D volume array
        volume_shape = (x_max - x_min, y_max - y_min, len(shapes))
        volume = np.zeros(volume_shape, dtype=np.float32)
        
        print(f"Creating volume of shape: {volume_shape}")
        
        # Fill the volume
        for i, shape in enumerate(shapes):
            # Get the actual coordinates from the original mask
            y_coords = shape['y_coords'] 
            x_coords = shape['x_coords']
            
            # Map to volume coordinates with proper offset
            vol_x = x_coords - x_min
            vol_y = y_coords - y_min
            
            # Ensure coordinates are within bounds
            valid_indices = (
                (vol_x >= 0) & (vol_x < volume_shape[0]) &
                (vol_y >= 0) & (vol_y < volume_shape[1])
            )
            
            vol_x = vol_x[valid_indices]
            vol_y = vol_y[valid_indices]
            
            if len(vol_x) > 0:
                volume[vol_x, vol_y, i] = 1.0
                
        print(f"Volume filled with {np.sum(volume)} voxels")
        
        return {
            'volume': volume,
            'shapes': shapes,
            'info': info,
            'bounds': (x_min, x_max, y_min, y_max, t_min, t_max),
            'downsample': downsample
        }
    
    def plot_3d_volume_matplotlib(self, worldline_idx, downsample=2, alpha=0.6):
        """Plot 3D volume using matplotlib with isosurfaces"""
        volume_data = self.build_3d_volume(worldline_idx, downsample)
        if volume_data is None:
            return
            
        volume = volume_data['volume']
        info = volume_data['info']
        
        print("Creating 3D visualization...")
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create isosurface for each time step
        colors = plt.cm.viridis(np.linspace(0, 1, volume.shape[2]))
        
        for t in range(volume.shape[2]):
            if np.sum(volume[:, :, t]) == 0:
                continue
                
            # Create 3D coordinates for this time slice
            x_coords, y_coords = np.where(volume[:, :, t] > 0.5)
            
            if len(x_coords) > 0:
                # Create voxel-like visualization
                z_coords = np.full_like(x_coords, t)
                
                ax.scatter(x_coords, y_coords, z_coords, 
                          c=[colors[t]], s=20, alpha=alpha, marker='s')
                
                # Optionally add contours at each level
                try:
                    # Create contour lines for this time slice
                    contours = measure.find_contours(volume[:, :, t], 0.5)
                    for contour in contours:
                        # Add some thickness to the contour
                        for offset in [-0.2, 0, 0.2]:
                            ax.plot(contour[:, 0], contour[:, 1], t + offset, 
                                   color=colors[t], alpha=alpha*0.7, linewidth=1)
                except:
                    pass  # Skip contours if they fail
        
        # Add trajectory line through centroids
        if len(info['centroid_x']) > 0:
            # Adjust centroids to volume coordinates
            bounds = volume_data['bounds']
            adj_x = (info['centroid_x'] - bounds[0]) / downsample
            adj_y = (info['centroid_y'] - bounds[2]) / downsample
            time_indices = np.arange(len(adj_x))
            
            ax.plot(adj_x, adj_y, time_indices, 'r-', linewidth=3, alpha=0.8, label='Centroid path')
            
            # Mark start and end
            ax.scatter([adj_x[0]], [adj_y[0]], [0], s=200, c='green', marker='^', 
                      edgecolors='darkgreen', linewidth=2, label='Start')
            ax.scatter([adj_x[-1]], [adj_y[-1]], [len(adj_x)-1], s=200, c='red', marker='v', 
                      edgecolors='darkred', linewidth=2, label='End')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_zlabel('Time Step')
        ax.set_title(f'3D Volume Evolution - Worldline {worldline_idx}\n'
                    f'{len(info["active_frames"])} frames, downsample={downsample}x')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def save_3d_model(self, worldline_idx, downsample=2, format='obj', output_file=None):
        """Save the 3D volume as a mesh file (OBJ, STL, or PLY)"""
        volume_data = self.build_3d_volume(worldline_idx, downsample)
        if volume_data is None:
            return
            
        volume = volume_data['volume']
        info = volume_data['info']
        
        if np.sum(volume) == 0:
            print("No volume data to save!")
            return
            
        print("Generating 3D mesh...")
        
        # Create marching cubes surface
        try:
            verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
        except Exception as e:
            print(f"Error generating mesh: {e}")
            print("Trying with different parameters...")
            # Try with smoothing
            smooth_volume = ndimage.gaussian_filter(volume, sigma=0.5)
            try:
                verts, faces, normals, values = measure.marching_cubes(smooth_volume, level=0.3)
            except:
                print("Could not generate mesh. Creating point cloud instead...")
                return self.save_point_cloud(worldline_idx, downsample, format, output_file)
        
        # Set default filename if not provided
        if output_file is None:
            output_file = f"worldline_{worldline_idx}_3d.{format.lower()}"
        
        print(f"Saving 3D model with {len(verts)} vertices and {len(faces)} faces...")
        
        if format.lower() == 'obj':
            self._save_obj(verts, faces, output_file)
        elif format.lower() == 'stl':
            self._save_stl(verts, faces, output_file)
        elif format.lower() == 'ply':
            self._save_ply(verts, faces, normals, output_file)
        else:
            print(f"Unsupported format: {format}")
            return
            
        print(f"✅ 3D model saved to {output_file}")
        
        # Print model stats
        print(f"\n📊 3D Model Statistics:")
        print(f"   Vertices: {len(verts):,}")
        print(f"   Faces: {len(faces):,}")
        print(f"   Volume bounds: {verts.min(axis=0)} to {verts.max(axis=0)}")
        print(f"   File format: {format.upper()}")
        
    def save_point_cloud(self, worldline_idx, downsample=2, format='obj', output_file=None):
        """Save as point cloud when mesh generation fails"""
        volume_data = self.build_3d_volume(worldline_idx, downsample)
        if volume_data is None:
            return
            
        volume = volume_data['volume']
        
        # Get voxel coordinates
        x_coords, y_coords, t_coords = np.where(volume > 0.5)
        
        if len(x_coords) == 0:
            print("No points to save!")
            return
            
        if output_file is None:
            output_file = f"worldline_{worldline_idx}_pointcloud.{format.lower()}"
        
        print(f"Saving point cloud with {len(x_coords)} points...")
        
        if format.lower() == 'obj':
            with open(output_file, 'w') as f:
                f.write("# Worldline Point Cloud\n")
                for x, y, t in zip(x_coords, y_coords, t_coords):
                    f.write(f"v {x} {y} {t}\n")
        elif format.lower() == 'ply':
            self._save_point_cloud_ply(x_coords, y_coords, t_coords, output_file)
        
        print(f"✅ Point cloud saved to {output_file}")
    
    def _save_obj(self, vertices, faces, filename):
        """Save mesh as OBJ file"""
        with open(filename, 'w') as f:
            f.write("# Worldline 3D Model\n")
            f.write(f"# Generated from storm tracking data\n\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def _save_stl(self, vertices, faces, filename):
        """Save mesh as STL file (ASCII)"""
        with open(filename, 'w') as f:
            f.write("solid WorldlineModel\n")
            
            for face in faces:
                # Calculate normal (though marching cubes should provide it)
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid WorldlineModel\n")
    
    def _save_ply(self, vertices, faces, normals, filename):
        """Save mesh as PLY file"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices with normals
            for i, vertex in enumerate(vertices):
                normal = normals[i] if i < len(normals) else [0, 0, 1]
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                       f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    def _save_point_cloud_ply(self, x_coords, y_coords, t_coords, filename):
        """Save point cloud as PLY file"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(x_coords)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Color by time
            colors = plt.cm.viridis(t_coords / t_coords.max())[:, :3] * 255
            
            for x, y, t, color in zip(x_coords, y_coords, t_coords, colors):
                f.write(f"{x:.6f} {y:.6f} {t:.6f} "
                       f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
        """Create interactive 3D volume using Plotly"""
        volume_data = self.build_3d_volume(worldline_idx, downsample)
        if volume_data is None:
            return
            
        volume = volume_data['volume']
        info = volume_data['info']
        
        print("Creating interactive 3D visualization...")
        
        # Extract all voxel coordinates
        x_coords, y_coords, t_coords = np.where(volume > 0.5)
        
        if len(x_coords) == 0:
            print("No voxels to display!")
            return
        
        # Color by time
        colors = t_coords.astype(float)
        
        # Create the 3D scatter plot
        fig = go.Figure()
        
        # Add voxels
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=t_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Time Step")
            ),
            name='Storm Volume',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Time: %{z}<extra></extra>'
        ))
        
        # Add centroid trajectory
        if len(info['centroid_x']) > 0:
            bounds = volume_data['bounds']
            adj_x = (info['centroid_x'] - bounds[0]) / downsample
            adj_y = (info['centroid_y'] - bounds[2]) / downsample
            time_indices = np.arange(len(adj_x))
            
            fig.add_trace(go.Scatter3d(
                x=adj_x,
                y=adj_y,
                z=time_indices,
                mode='lines+markers',
                line=dict(color='red', width=6),
                marker=dict(size=5, color='red'),
                name='Centroid Path',
                hovertemplate='Time: %{z}<br>X: %{customdata[0]:.1f}<br>Y: %{customdata[1]:.1f}<br>Area: %{customdata[2]:.0f}<extra></extra>',
                customdata=np.column_stack([info['centroid_x'], info['centroid_y'], info['area']])
            ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter3d(
                x=[adj_x[0]],
                y=[adj_y[0]], 
                z=[0],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[adj_x[-1]],
                y=[adj_y[-1]],
                z=[len(adj_x)-1], 
                mode='markers',
                marker=dict(size=10, color='darkred', symbol='diamond'),
                name='End'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Interactive 3D Storm Evolution - Worldline {worldline_idx}<br>'
                  f'<sub>{len(info["active_frames"])} frames, downsample={downsample}x</sub>',
            scene=dict(
                xaxis_title='X Position (pixels)',
                yaxis_title='Y Position (pixels)',
                zaxis_title='Time Step',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"💾 Saved interactive plot to {save_html}")
        
        fig.show()
        
    def plot_3d_volume_plotly(self, worldline_idx, downsample=3, save_html=None):
        """Create interactive 3D volume using Plotly"""
        volume_data = self.build_3d_volume(worldline_idx, downsample)
        if volume_data is None:
            return
            
        volume = volume_data['volume']
        info = volume_data['info']
        
        print("Creating interactive 3D visualization...")
        
        # Extract all voxel coordinates
        x_coords, y_coords, t_coords = np.where(volume > 0.5)
        
        print(f"Found {len(x_coords)} voxels to display")
        
        if len(x_coords) == 0:
            print("No voxels to display!")
            return
        
        # Color by time
        colors = t_coords.astype(float)
        
        # Create the 3D scatter plot
        fig = go.Figure()
        
        # Add voxels
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=t_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Time Step")
            ),
            name='Storm Volume',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Time: %{z}<extra></extra>'
        ))
        
        # Add centroid trajectory
        if len(info['centroid_x']) > 0:
            bounds = volume_data['bounds']
            adj_x = (info['centroid_x'] - bounds[0]) / downsample
            adj_y = (info['centroid_y'] - bounds[2]) / downsample
            time_indices = np.arange(len(adj_x))
            
            fig.add_trace(go.Scatter3d(
                x=adj_x,
                y=adj_y,
                z=time_indices,
                mode='lines+markers',
                line=dict(color='red', width=6),
                marker=dict(size=5, color='red'),
                name='Centroid Path',
                hovertemplate='Time: %{z}<br>X: %{customdata[0]:.1f}<br>Y: %{customdata[1]:.1f}<br>Area: %{customdata[2]:.0f}<extra></extra>',
                customdata=np.column_stack([info['centroid_x'], info['centroid_y'], info['area']])
            ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter3d(
                x=[adj_x[0]],
                y=[adj_y[0]], 
                z=[0],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[adj_x[-1]],
                y=[adj_y[-1]],
                z=[len(adj_x)-1], 
                mode='markers',
                marker=dict(size=10, color='darkred', symbol='diamond'),
                name='End'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Interactive 3D Storm Evolution - Worldline {worldline_idx}<br>'
                  f'<sub>{len(info["active_frames"])} frames, downsample={downsample}x</sub>',
            scene=dict(
                xaxis_title='X Position (pixels)',
                yaxis_title='Y Position (pixels)',
                zaxis_title='Time Step',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"💾 Saved interactive plot to {save_html}")
        
        fig.show()
        
        # Print volume statistics
        total_voxels = np.sum(volume > 0.5)
        print(f"\n📊 3D Volume Statistics:")
        print(f"   Total voxels: {total_voxels:,}")
        print(f"   Volume shape: {volume.shape}")
        print(f"   Density per frame: {total_voxels/volume.shape[2]:.1f} voxels/frame")
        print(f"   Downsample factor: {downsample}x")

def main():
    parser = argparse.ArgumentParser(description='Create 3D volumetric visualization of worldline shapes')
    parser.add_argument('--worldline', type=int, default=None,
                       help='Worldline index to visualize')
    parser.add_argument('--worldline-file', type=str, default='worldline_tensor.pt',
                       help='Path to worldline tensor file')
    parser.add_argument('--npy-dir', type=str, default='./unpacked_npys',
                       help='Directory containing NPY mask files')
    parser.add_argument('--downsample', type=int, default=2,
                       help='Downsample factor for performance (1=full res, 2=half, etc.)')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive Plotly visualization instead of matplotlib')
    parser.add_argument('--save-html', type=str, default=None,
                       help='Save interactive plot as HTML file')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Transparency for matplotlib plot')
    
    # 3D Model export options
    parser.add_argument('--save-3d', type=str, default=None,
                       help='Save 3D model to file (specify filename)')
    parser.add_argument('--format', choices=['obj', 'stl', 'ply'], default='obj',
                       help='3D model format (obj, stl, or ply)')
    
    args = parser.parse_args()
    
    viz = Worldline3DVolumeVisualizer(args.worldline_file, args.npy_dir)
    
    if args.worldline is None:
        # Show suggestions
        lengths = torch.sum(viz.existence_mask, dim=1).numpy()
        top_indices = np.argsort(lengths)[-10:][::-1]
        
        print("🔍 Top 10 Longest Worldlines for 3D Visualization:")
        print("=" * 60)
        
        for rank, idx in enumerate(top_indices):
            info = viz.get_worldline_info(idx)
            if info is not None:
                print(f"{rank+1:2d}. Worldline {idx:4d}: Length={len(info['active_frames']):2d} frames")
        
        print(f"\n💡 Usage Examples:")
        print(f"   # Matplotlib 3D volume")
        print(f"   python {__file__} --worldline {top_indices[0]} --downsample 2")
        print(f"   # Interactive Plotly version") 
        print(f"   python {__file__} --worldline {top_indices[0]} --interactive")
        print(f"   # Save as 3D model")
        print(f"   python {__file__} --worldline {top_indices[0]} --save-3d storm.obj")
        print(f"   python {__file__} --worldline {top_indices[0]} --save-3d storm.stl --format stl")
        print(f"   # Save as HTML")
        print(f"   python {__file__} --worldline {top_indices[0]} --interactive --save-html storm.html")
        return
    
    # Save 3D model if requested
    if args.save_3d:
        viz.save_3d_model(args.worldline, args.downsample, args.format, args.save_3d)
    
    # Show visualization
    if args.interactive:
        viz.plot_3d_volume_plotly(args.worldline, args.downsample, args.save_html)
    else:
        viz.plot_3d_volume_matplotlib(args.worldline, args.downsample, args.alpha)

if __name__ == "__main__":
    main()