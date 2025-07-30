import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from collections import defaultdict
import cv2

class WorldlineBuilder:
    def __init__(self, max_distance=50.0, min_overlap_ratio=0.3, max_gap_frames=2):
        """
        Parameters:
        - max_distance: Max pixel distance for centroid matching
        - min_overlap_ratio: Min IoU for considering masks as same worldline
        - max_gap_frames: Max frames a worldline can be missing before considered ended
        """
        self.max_distance = max_distance
        self.min_overlap_ratio = min_overlap_ratio
        self.max_gap_frames = max_gap_frames
        
    def extract_features(self, mask, instance_id):
        """Extract geometric features from a single instance mask"""
        instance_mask = (mask == instance_id).astype(np.uint8)
        
        if np.sum(instance_mask) == 0:
            return None
            
        # Get region properties
        props = regionprops(instance_mask)[0]
        
        # Basic geometric features
        centroid = props.centroid  # (y, x)
        area = props.area
        perimeter = props.perimeter
        bbox = props.bbox  # (min_row, min_col, max_row, max_col)
        
        # Shape features
        major_axis = props.major_axis_length
        minor_axis = props.minor_axis_length
        orientation = props.orientation
        eccentricity = props.eccentricity
        
        # Bounding box features
        bbox_width = bbox[3] - bbox[1]
        bbox_height = bbox[2] - bbox[0]
        aspect_ratio = bbox_width / max(bbox_height, 1e-6)
        
        # Compactness (area vs perimeter)
        compactness = 4 * np.pi * area / max(perimeter**2, 1e-6)
        
        features = np.array([
            centroid[1],  # x centroid
            centroid[0],  # y centroid
            area,
            perimeter,
            bbox_width,
            bbox_height,
            aspect_ratio,
            major_axis,
            minor_axis,
            orientation,
            eccentricity,
            compactness
        ], dtype=np.float32)
        
        return {
            'features': features,
            'centroid': centroid,
            'bbox': bbox,
            'mask': instance_mask
        }
    
    def compute_iou(self, mask1, mask2):
        """Compute intersection over union of two binary masks"""
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        return intersection / max(union, 1e-6)
    
    def associate_instances(self, prev_instances, curr_instances):
        """Associate instances between consecutive frames using Hungarian algorithm"""
        if len(prev_instances) == 0 or len(curr_instances) == 0:
            return {}
            
        # Compute cost matrix based on centroid distance and IoU
        costs = np.zeros((len(prev_instances), len(curr_instances)))
        
        prev_centroids = np.array([inst['centroid'] for inst in prev_instances.values()])
        curr_centroids = np.array([inst['centroid'] for inst in curr_instances.values()])
        
        # Distance cost
        dist_matrix = cdist(prev_centroids, curr_centroids)
        
        prev_ids = list(prev_instances.keys())
        curr_ids = list(curr_instances.keys())
        
        for i, prev_id in enumerate(prev_ids):
            for j, curr_id in enumerate(curr_ids):
                distance = dist_matrix[i, j]
                
                # If too far, make cost very high
                if distance > self.max_distance:
                    costs[i, j] = 1e6
                    continue
                
                # Compute IoU
                iou = self.compute_iou(
                    prev_instances[prev_id]['mask'],
                    curr_instances[curr_id]['mask']
                )
                
                # Combined cost: distance + (1 - IoU)
                costs[i, j] = distance / self.max_distance + (1 - iou)
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(costs)
        
        associations = {}
        for i, j in zip(row_ind, col_ind):
            if costs[i, j] < 1e5:  # Valid association
                prev_id = prev_ids[i]
                curr_id = curr_ids[j]
                
                # Check minimum overlap requirement
                iou = self.compute_iou(
                    prev_instances[prev_id]['mask'],
                    curr_instances[curr_id]['mask']
                )
                
                if iou >= self.min_overlap_ratio:
                    associations[curr_id] = prev_id
        
        return associations
    
    def build_worldlines(self, masks):
        """Build worldlines from sequence of instance masks"""
        worldlines = {}  # worldline_id -> {frames: {frame_idx: features}, masks: {frame_idx: mask}, last_seen: frame_idx}
        next_worldline_id = 0
        instance_to_worldline = {}  # (frame_idx, instance_id) -> worldline_id
        
        print("Building worldlines...")
        
        for frame_idx in tqdm(range(len(masks)), desc="Processing frames", unit="frame"):
            mask = masks[frame_idx]
            
            # Extract instances from current frame
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background
            
            curr_instances = {}
            for instance_id in unique_ids:
                features = self.extract_features(mask, instance_id)
                if features is not None:
                    curr_instances[instance_id] = features
            
            if frame_idx == 0:
                # Initialize worldlines for first frame
                for instance_id in curr_instances:
                    worldlines[next_worldline_id] = {
                        'frames': {frame_idx: curr_instances[instance_id]['features']},
                        'masks': {frame_idx: curr_instances[instance_id]['mask']},  # Store mask
                        'last_seen': frame_idx,
                        'gaps': 0
                    }
                    instance_to_worldline[(frame_idx, instance_id)] = next_worldline_id
                    next_worldline_id += 1
            else:
                # Get previous frame instances that are still active
                prev_instances = {}
                for (prev_frame, prev_inst_id), wl_id in instance_to_worldline.items():
                    if prev_frame == frame_idx - 1 and wl_id in worldlines:
                        if worldlines[wl_id]['gaps'] <= self.max_gap_frames:
                            # Reconstruct previous instance features
                            prev_features = worldlines[wl_id]['frames'][prev_frame]
                            prev_mask = worldlines[wl_id]['masks'][prev_frame]  # Get stored mask
                            prev_instances[prev_inst_id] = {
                                'centroid': (prev_features[1], prev_features[0]),  # (y, x)
                                'features': prev_features,
                                'mask': prev_mask,  # Include mask
                                'worldline_id': wl_id
                            }
                
                # Associate current instances with previous ones
                if len(prev_instances) > 0 and len(curr_instances) > 0:
                    print(f"  Frame {frame_idx}: Associating {len(curr_instances)} instances with {len(prev_instances)} previous...")
                    associations = self.associate_instances(prev_instances, curr_instances)
                else:
                    associations = {}
                
                # Update existing worldlines
                for curr_id, prev_id in associations.items():
                    wl_id = prev_instances[prev_id]['worldline_id']
                    worldlines[wl_id]['frames'][frame_idx] = curr_instances[curr_id]['features']
                    worldlines[wl_id]['masks'][frame_idx] = curr_instances[curr_id]['mask']  # Store mask
                    worldlines[wl_id]['last_seen'] = frame_idx
                    worldlines[wl_id]['gaps'] = 0
                    instance_to_worldline[(frame_idx, curr_id)] = wl_id
                
                # Create new worldlines for unassociated instances
                unassociated = set(curr_instances.keys()) - set(associations.keys())
                for instance_id in unassociated:
                    worldlines[next_worldline_id] = {
                        'frames': {frame_idx: curr_instances[instance_id]['features']},
                        'masks': {frame_idx: curr_instances[instance_id]['mask']},  # Store mask
                        'last_seen': frame_idx,
                        'gaps': 0
                    }
                    instance_to_worldline[(frame_idx, instance_id)] = next_worldline_id
                    next_worldline_id += 1
                
                # Increment gaps for worldlines not seen in current frame
                for wl_id in worldlines:
                    if worldlines[wl_id]['last_seen'] < frame_idx:
                        worldlines[wl_id]['gaps'] += 1
        
        return worldlines
    
    def pack_worldlines_to_tensor(self, worldlines, num_frames, feature_dim=12):
        """Pack worldlines into a tensor format"""
        print("Filtering and packing worldlines...")
        
        # Filter out very short worldlines (< 3 frames)
        valid_worldlines = {
            wl_id: wl for wl_id, wl in tqdm(worldlines.items(), desc="Filtering worldlines") 
            if len(wl['frames']) >= 3
        }
        
        if not valid_worldlines:
            print("No valid worldlines found!")
            return None, None
        
        print(f"Found {len(valid_worldlines)} valid worldlines")
        
        # Create tensor: (num_worldlines, num_frames, feature_dim)
        num_worldlines = len(valid_worldlines)
        worldline_tensor = torch.zeros((num_worldlines, num_frames, feature_dim))
        existence_mask = torch.zeros((num_worldlines, num_frames), dtype=torch.bool)
        
        worldline_ids = list(valid_worldlines.keys())
        
        for wl_idx, wl_id in tqdm(enumerate(worldline_ids), total=len(worldline_ids), desc="Packing worldlines"):
            worldline = valid_worldlines[wl_id]
            
            for frame_idx, features in worldline['frames'].items():
                worldline_tensor[wl_idx, frame_idx] = torch.from_numpy(features)
                existence_mask[wl_idx, frame_idx] = True
        
        # Compute temporal features (velocity, acceleration)
        enhanced_tensor = self.add_temporal_features(worldline_tensor, existence_mask)
        
        return enhanced_tensor, existence_mask
    
    def add_temporal_features(self, worldline_tensor, existence_mask):
        """Add velocity and acceleration features"""
        print("Computing temporal features (velocity, acceleration)...")
        num_worldlines, num_frames, feature_dim = worldline_tensor.shape
        
        # Extract centroids (first 2 features)
        centroids = worldline_tensor[:, :, :2]  # (num_worldlines, num_frames, 2)
        
        # Compute velocities
        velocities = torch.zeros_like(centroids)
        velocities[:, 1:] = centroids[:, 1:] - centroids[:, :-1]
        
        # Compute accelerations
        accelerations = torch.zeros_like(centroids)
        accelerations[:, 1:] = velocities[:, 1:] - velocities[:, :-1]
        
        # Compute area change rate
        areas = worldline_tensor[:, :, 2:3]  # (num_worldlines, num_frames, 1)
        area_change = torch.zeros_like(areas)
        area_change[:, 1:] = areas[:, 1:] - areas[:, :-1]
        
        # Concatenate all features
        enhanced_features = torch.cat([
            worldline_tensor,  # Original 12 features
            velocities,        # 2 velocity features
            accelerations,     # 2 acceleration features  
            area_change        # 1 area change feature
        ], dim=-1)
        
        # Zero out features where worldline doesn't exist
        enhanced_features = enhanced_features * existence_mask.unsqueeze(-1)
        
        return enhanced_features

def main():
    # Your folder configuration
    FOLDERS = [
        '/home/mwigder/NEXRAD_Subset/June/Week_1_savepath',
        '/home/mwigder/NEXRAD_Subset/June/Week_2_savepath',
        '/home/mwigder/NEXRAD_Subset/June/Week_3_savepath',
        '/home/mwigder/NEXRAD_Subset/June/Week_4_savepath',
        '/home/mwigder/NEXRAD_Subset/July/Week_1_savepath',
        '/home/mwigder/NEXRAD_Subset/July/Week_2_savepath',
        '/home/mwigder/NEXRAD_Subset/July/Week_3_savepath',
        '/home/mwigder/NEXRAD_Subset/July/Week_4_savepath',
        '/home/mwigder/NEXRAD_Subset/August/Week_1_savepath',
        '/home/mwigder/NEXRAD_Subset/August/Week_2_savepath',
        '/home/mwigder/NEXRAD_Subset/August/Week_3_savepath',
        '/home/mwigder/NEXRAD_Subset/August/Week_4_savepath',
    ]
    
    # Step 1: Check if NPY files already exist, if not unpack pickle files
    NpyOutputDir = './unpacked_npys'
    META_CSV = os.path.join(NpyOutputDir, "mask_index.csv")
    
    if os.path.exists(META_CSV) and os.path.exists(NpyOutputDir):
        # Check if we have NPY files
        existing_npys = [f for f in os.listdir(NpyOutputDir) if f.endswith('.npy')]
        if len(existing_npys) > 0:
            print(f"Step 1/5: Found existing NPY files ({len(existing_npys)} files)")
            print("Skipping unpacking step. Delete './unpacked_npys' folder to force re-processing.")
        else:
            print("Step 1/5: NPY directory exists but is empty, unpacking...")
            unpack_pickle_files(FOLDERS, NpyOutputDir, META_CSV)
    else:
        print("Step 1/5: No existing NPY files found, unpacking pickle files...")
        os.makedirs(NpyOutputDir, exist_ok=True)
        unpack_pickle_files(FOLDERS, NpyOutputDir, META_CSV)
    
    # Step 2: Load all masks in temporal order
    print("Step 2/5: Loading masks...")
    meta = pd.read_csv(META_CSV)
    
    mask_list = []
    timestamps = []
    
    for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading mask files"):
        try:
            mask = np.load(os.path.join(NpyOutputDir, row['npy_file']))
            mask_list.append(mask)
            timestamps.append(row['timestamp'])
        except Exception as e:
            print(f"Error loading {row['npy_file']}: {e}")
            continue
    
    if not mask_list:
        print("No masks loaded successfully!")
        return
    
    # Step 3: Build worldlines
    print(f"Step 3/5: Building worldlines from {len(mask_list)} masks...")
    builder = WorldlineBuilder()
    worldlines = builder.build_worldlines(mask_list)
    
    # Step 4: Pack into tensor
    print("Step 4/5: Packing worldlines into tensor...")
    worldline_tensor, existence_mask = builder.pack_worldlines_to_tensor(
        worldlines, len(mask_list)
    )
    
    if worldline_tensor is None:
        print("Failed to create worldline tensor!")
        return
    
    # Step 5: Save results
    print("Step 5/5: Saving results...")
    print(f"Final worldline tensor shape: {worldline_tensor.shape}")
    print(f"Final existence mask shape: {existence_mask.shape}")
    
    output_data = {
        'worldlines': worldline_tensor,
        'existence_mask': existence_mask,
        'timestamps': timestamps,
        'feature_names': [
            'centroid_x', 'centroid_y', 'area', 'perimeter', 
            'bbox_width', 'bbox_height', 'aspect_ratio',
            'major_axis', 'minor_axis', 'orientation', 
            'eccentricity', 'compactness',
            'velocity_x', 'velocity_y',
            'acceleration_x', 'acceleration_y',
            'area_change'
        ]
    }
    
    torch.save(output_data, "worldline_tensor.pt")
    print("✅ Saved worldline_tensor.pt!")
    
    # Print some statistics
    valid_worldlines = torch.sum(existence_mask, dim=1)
    print(f"\n📊 Worldline Statistics:")
    print(f"  Total worldlines: {len(valid_worldlines)}")
    print(f"  Worldline lengths: min={valid_worldlines.min()}, max={valid_worldlines.max()}, mean={valid_worldlines.float().mean():.1f}")
    print(f"  Total frames: {len(mask_list)}")
    print(f"  Tensor size: {worldline_tensor.numel() * 4 / 1024**2:.1f} MB")



def unpack_pickle_files(FOLDERS, NpyOutputDir, META_CSV):
    """Separate function to handle pickle unpacking"""
    meta_file = open(META_CSV, "w")
    meta_file.write("npy_file,timestamp\n")
    
    total_files = sum(len([f for f in os.listdir(folder) if f.endswith(".pickle")]) 
                     for folder in FOLDERS if os.path.exists(folder))
    print(f"Found {total_files} pickle files to process")
    
    file_counter = 0
    with tqdm(total=total_files, desc="Processing pickle files") as pbar:
        
        for folder in FOLDERS:
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist, skipping...")
                continue
                
            file_list = sorted([f for f in os.listdir(folder) if f.endswith(".pickle")])
            
            for file in file_list:
                file_path = os.path.join(folder, file)
                
                try:
                    with open(file_path, 'rb') as fp:
                        ds = pickle.load(fp)
                        masks = ds['Cell_Segmentation'].values
                        times = ds['time'].values
                        
                        for i in range(masks.shape[0]):
                            mask = masks[i]
                            dt = times[i]
                            dt_str = str(dt) if isinstance(dt, np.datetime64) else dt
                            out_fname = f"{os.path.splitext(file)[0]}_t{i:04d}.npy"
                            np.save(os.path.join(NpyOutputDir, out_fname), mask)
                            meta_file.write(f"{out_fname},{dt_str}\n")
                            
                    file_counter += 1
                    pbar.update(1)
                    pbar.set_postfix(folder=os.path.basename(folder), file=file)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    pbar.update(1)
                    continue
    
    meta_file.close()
    print(f"Processed {file_counter} files successfully")


if __name__ == "__main__":
    main()

