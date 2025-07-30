#!/usr/bin/env python3
"""
Universal tensor packing script for radar data and masks.
Handles various data organization patterns and file formats.
"""

import os
import re
import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union


class TensorPacker:
    """Flexible tensor packer for different data organization patterns"""
    
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
    
    def get_sorted_dirs(self, base_dir: str, prefix: str) -> List[str]:
        """Get directories matching prefix pattern, sorted by index"""
        dirs = []
        for d in os.listdir(base_dir):
            m = re.match(f"{prefix}(\\d+)$", d)
            if m and os.path.isdir(os.path.join(base_dir, d)):
                idx = int(m.group(1))
                dirs.append((idx, os.path.join(base_dir, d)))
        dirs.sort()
        return [path for _, path in dirs]
    
    def load_file(self, filepath: str) -> np.ndarray:
        """Load data from various file formats"""
        ext = Path(filepath).suffix.lower()
        
        if ext == '.npy':
            return np.load(filepath)
        elif ext == '.npz':
            data = np.load(filepath)
            # If npz has multiple arrays, take the first one or 'data' key
            if 'data' in data:
                return data['data']
            else:
                return data[list(data.keys())[0]]
        elif ext in ['.pkl', '.pickle']:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif ext == '.pt':
            return torch.load(filepath).numpy()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def find_matching_files(self, radar_dir: str, mask_dir: str, 
                           radar_pattern: str = "*.npy", 
                           mask_pattern: str = "*.npy") -> List[Tuple[str, str]]:
        """Find matching radar and mask files"""
        from glob import glob
        
        radar_files = sorted(glob(os.path.join(radar_dir, radar_pattern)))
        mask_files = sorted(glob(os.path.join(mask_dir, mask_pattern)))
        
        if len(radar_files) != len(mask_files):
            print(f"Warning: {len(radar_files)} radar files vs {len(mask_files)} mask files")
            # Try to match by filename
            radar_names = {Path(f).stem: f for f in radar_files}
            mask_names = {Path(f).stem: f for f in mask_files}
            
            common_names = set(radar_names.keys()) & set(mask_names.keys())
            if common_names:
                matched_pairs = [(radar_names[name], mask_names[name]) for name in sorted(common_names)]
                print(f"Matched {len(matched_pairs)} files by name")
                return matched_pairs
        
        # Default: pair by order
        min_len = min(len(radar_files), len(mask_files))
        return list(zip(radar_files[:min_len], mask_files[:min_len]))
    
    def pack_indexed_directories(self, base_dir: str, out_path: str, 
                               radar_prefix: str = "processed_in", 
                               mask_prefix: str = "processed_out") -> None:
        """Pack data from indexed directories (processed_in0, processed_out0, etc.)"""
        
        radar_dirs = self.get_sorted_dirs(base_dir, radar_prefix)
        mask_dirs = self.get_sorted_dirs(base_dir, mask_prefix)
        
        if len(radar_dirs) != len(mask_dirs):
            raise ValueError(f"Found {len(radar_dirs)} {radar_prefix} dirs but {len(mask_dirs)} {mask_prefix} dirs")
        
        # Create index maps
        radar_map = {int(re.search(r'(\d+)$', d).group(1)): d for d in radar_dirs}
        mask_map = {int(re.search(r'(\d+)$', d).group(1)): d for d in mask_dirs}
        
        # Find common indices
        common_indices = sorted(set(radar_map.keys()) & set(mask_map.keys()))
        if not common_indices:
            raise RuntimeError("No matching index folders found.")
        
        print(f"Found {len(common_indices)} matching directory pairs")
        
        # Count total files
        total_files = 0
        for idx in common_indices:
            radar_dir = radar_map[idx]
            total_files += len([f for f in os.listdir(radar_dir) 
                              if Path(f).suffix.lower() in ['.npy', '.npz', '.pkl', '.pickle']])
        
        # Pack all files
        all_radar, all_masks = [], []
        pbar = tqdm(total=total_files, desc="Packing indexed directories")
        
        for idx in common_indices:
            radar_dir = radar_map[idx]
            mask_dir = mask_map[idx]
            
            file_pairs = self.find_matching_files(radar_dir, mask_dir)
            
            for radar_file, mask_file in file_pairs:
                radar_data = self.load_file(radar_file)
                mask_data = self.load_file(mask_file)
                
                all_radar.append(radar_data)
                all_masks.append(mask_data)
                pbar.update(1)
        
        pbar.close()
        self._save_tensors(all_radar, all_masks, out_path)
    
    def pack_simple_directories(self, radar_dir: str, mask_dir: str, out_path: str,
                              radar_pattern: str = "*.npy", mask_pattern: str = "*.npy") -> None:
        """Pack data from two simple directories"""
        
        file_pairs = self.find_matching_files(radar_dir, mask_dir, radar_pattern, mask_pattern)
        
        if not file_pairs:
            raise ValueError("No matching files found")
        
        print(f"Found {len(file_pairs)} matching file pairs")
        
        all_radar, all_masks = [], []
        pbar = tqdm(file_pairs, desc="Packing files")
        
        for radar_file, mask_file in pbar:
            pbar.set_postfix({'file': Path(radar_file).name})
            
            radar_data = self.load_file(radar_file)
            mask_data = self.load_file(mask_file)
            
            all_radar.append(radar_data)
            all_masks.append(mask_data)
        
        self._save_tensors(all_radar, all_masks, out_path)
    
    def pack_mixed_formats(self, file_pairs: List[Tuple[str, str]], out_path: str) -> None:
        """Pack data from a list of (radar_file, mask_file) pairs"""
        
        all_radar, all_masks = [], []
        pbar = tqdm(file_pairs, desc="Packing mixed formats")
        
        for radar_file, mask_file in pbar:
            pbar.set_postfix({'radar': Path(radar_file).name, 'mask': Path(mask_file).name})
            
            radar_data = self.load_file(radar_file)
            mask_data = self.load_file(mask_file)
            
            all_radar.append(radar_data)
            all_masks.append(mask_data)
        
        self._save_tensors(all_radar, all_masks, out_path)
    
    def _save_tensors(self, radar_data: List[np.ndarray], mask_data: List[np.ndarray], out_path: str) -> None:
        """Convert to tensors and save"""
        
        if len(radar_data) != len(mask_data):
            raise ValueError(f"Mismatch: {len(radar_data)} radar vs {len(mask_data)} mask samples")
        
        # Check shape consistency
        radar_shapes = {arr.shape for arr in radar_data}
        mask_shapes = {arr.shape for arr in mask_data}
        
        if len(radar_shapes) > 1:
            print(f"Warning: Multiple radar shapes found: {radar_shapes}")
        if len(mask_shapes) > 1:
            print(f"Warning: Multiple mask shapes found: {mask_shapes}")
        
        # Convert to tensors
        print("Converting to tensors...")
        try:
            radar_tensor = torch.from_numpy(np.stack(radar_data, axis=0)).to(self.dtype)
            mask_tensor = torch.from_numpy(np.stack(mask_data, axis=0)).to(self.dtype)
        except ValueError as e:
            print(f"Error stacking arrays: {e}")
            print("This usually means arrays have different shapes. Consider preprocessing.")
            raise
        
        # Save
        os.makedirs(Path(out_path).parent, exist_ok=True)
        torch.save({
            "inputs": radar_tensor, 
            "masks": mask_tensor
        }, out_path, _use_new_zipfile_serialization=True)
        
        print(f"✓ Saved {radar_tensor.shape[0]} samples to {out_path}")
        print(f"  Radar tensor shape: {radar_tensor.shape}")
        print(f"  Mask tensor shape: {mask_tensor.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pack radar and mask data into tensors")
    
    # Output
    parser.add_argument("--output", "-o", required=True, help="Output tensor file path")
    
    # Data organization modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indexed_dirs", help="Base directory with indexed subdirs (processed_in0, processed_out0, etc.)")
    group.add_argument("--simple_dirs", nargs=2, metavar=("RADAR_DIR", "MASK_DIR"), 
                      help="Two directories: radar_dir mask_dir")
    group.add_argument("--file_list", help="Text file with 'radar_file mask_file' pairs (one per line)")
    
    # Patterns and prefixes
    parser.add_argument("--radar_prefix", default="processed_in", 
                       help="Prefix for radar directories (for indexed mode)")
    parser.add_argument("--mask_prefix", default="processed_out", 
                       help="Prefix for mask directories (for indexed mode)")
    parser.add_argument("--radar_pattern", default="*.npy", 
                       help="File pattern for radar files")
    parser.add_argument("--mask_pattern", default="*.npy", 
                       help="File pattern for mask files")
    
    # Options
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "int32", "int64"],
                       help="Tensor data type")
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16, 
        "int32": torch.int32,
        "int64": torch.int64
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize packer
    packer = TensorPacker(dtype=dtype)
    
    try:
        if args.indexed_dirs:
            print(f"Packing indexed directories from: {args.indexed_dirs}")
            packer.pack_indexed_directories(
                args.indexed_dirs, args.output, 
                args.radar_prefix, args.mask_prefix
            )
        
        elif args.simple_dirs:
            radar_dir, mask_dir = args.simple_dirs
            print(f"Packing simple directories: {radar_dir} -> {mask_dir}")
            packer.pack_simple_directories(
                radar_dir, mask_dir, args.output,
                args.radar_pattern, args.mask_pattern
            )
        
        elif args.file_list:
            print(f"Packing from file list: {args.file_list}")
            with open(args.file_list, 'r') as f:
                file_pairs = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            file_pairs.append((parts[0], parts[1]))
            
            if not file_pairs:
                raise ValueError("No valid file pairs found in file list")
            
            packer.pack_mixed_formats(file_pairs, args.output)
        
        print("\n✓ Tensor packing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage:
    # python pack_tensor_data.py --indexed_dirs /path/to/training_data --output tensorstore.pt
    # python pack_tensor_data.py --simple_dirs /path/to/radar /path/to/masks --output tensorstore.pt
    # python pack_tensor_data.py --file_list file_pairs.txt --output tensorstore.pt
    
    exit(main())