#!/usr/bin/env python3
"""
Universal segmentation script for radar data.
Can process single or multiple time indices with configurable parameters.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from typing import List, Tuple, Optional, Union
import argparse
import pickle

# Import your model (adjust import path as needed)
from multiclass_model_v4 import create_model, Config


class RadarSegmenter:
    def __init__(self, model_checkpoint_path: str, config: Optional[Config] = None):
        """Initialize the segmenter with model and config."""
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = create_model(self.config)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Default normalization values (adjust as needed)
        self.train_min = 0.02
        self.train_max = 1.04
    
    def load_and_preprocess(self, file_path: str, variable_name: str, time_index: int) -> torch.Tensor:
        """Load and preprocess a single time slice from NetCDF file."""
        with Dataset(file_path, "r") as ds:
            if time_index >= ds.variables[variable_name].shape[0]:
                raise ValueError(f"Time index {time_index} out of range (max: {ds.variables[variable_name].shape[0]-1})")
            
            img_np = ds.variables[variable_name][time_index, :, :].astype(np.float32)
        
        # Clean and normalize
        img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
        img_np = np.clip(img_np, self.train_min, self.train_max)
        img_np = (img_np - self.train_min) / (self.train_max - self.train_min)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)
        
        return img_tensor
    
    def predict_single(self, img_tensor: torch.Tensor) -> dict:
        """Run prediction on a single image tensor."""
        with torch.no_grad():
            preds = self.model([img_tensor.to(self.device)])
            predictions = [{k: v.cpu() for k, v in p.items()} for p in preds]
        return predictions[0]
    
    def tile_and_predict(self, img_tensor: torch.Tensor, n_rows: int = 3, n_cols: int = 3) -> Tuple[List[dict], List[tuple]]:
        """Tile image and predict on each tile."""
        img_np = img_tensor.squeeze(0).numpy()
        tiles, coords = self._tile_image(img_np, n_rows, n_cols)
        
        tile_predictions = []
        for tile in tiles:
            tile_tensor = torch.from_numpy(tile).unsqueeze(0)
            pred = self.predict_single(tile_tensor)
            tile_predictions.append(pred)
        
        return tile_predictions, coords
    
    def _tile_image(self, img_np: np.ndarray, nrows: int, ncols: int) -> Tuple[List[np.ndarray], List[tuple]]:
        """Split image into tiles."""
        H, W = img_np.shape
        hsize, wsize = H // nrows, W // ncols
        tiles = []
        coords = []
        
        for i in range(nrows):
            for j in range(ncols):
                y0, y1 = i * hsize, (i + 1) * hsize if i < nrows - 1 else H
                x0, x1 = j * wsize, (j + 1) * wsize if j < ncols - 1 else W
                tiles.append(img_np[y0:y1, x0:x1])
                coords.append((y0, y1, x0, x1))
        
        return tiles, coords
    
    def visualize_predictions(self, predictions: dict, title: str = "Predicted Instances", 
                            alpha: float = 0.5, save_path: Optional[str] = None):
        """Visualize prediction masks."""
        masks = predictions['masks']
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        N = masks.shape[0]
        if N == 0:
            print(f"No instances found for {title}")
            return
        
        H, W = masks.shape[1], masks.shape[2]
        out_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Overlay each instance with random color
        for i in range(N):
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            mask_np = masks[i].numpy()
            out_img[mask_np > 0.9] = color
        
        plt.figure(figsize=(10, 10))
        plt.imshow(out_img, interpolation='nearest', origin="lower")
        plt.title(f"{title} (instances: {N})")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def stitch_tiled_predictions(self, tile_predictions: List[dict], tile_coords: List[tuple], 
                               canvas_shape: Tuple[int, int]) -> np.ndarray:
        """Stitch tiled predictions into full canvas."""
        canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8)
        
        for tile_pred, (y0, y1, x0, x1) in zip(tile_predictions, tile_coords):
            masks = tile_pred['masks']
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            N = masks.shape[0]
            for i in range(N):
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                mask_np = masks[i].numpy()
                mask_binary = mask_np > 0.9
                canvas[y0:y1, x0:x1][mask_binary] = color
        
        return canvas
    
    def _save_prediction_to_pickle(self, prediction: dict, pickle_path: str) -> None:
        """Save a prediction dict to pickle file"""
        import pickle
        
        # Convert tensors to numpy for pickling
        pickle_data = {
            'masks': prediction['masks'].numpy(),
            'boxes': prediction['boxes'].numpy(),
            'labels': prediction['labels'].numpy(),
            'scores': prediction['scores'].numpy()
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
    
    def _save_masks_to_pickle(self, tile_predictions: List[dict], tile_coords: List[tuple], pickle_path: str) -> None:
        """Save tiled predictions to pickle file"""
        import pickle
        
        pickle_data = {
            'tile_predictions': [],
            'tile_coords': tile_coords
        }
        
        # Convert each tile prediction to numpy
        for tile_pred in tile_predictions:
            tile_data = {
                'masks': tile_pred['masks'].numpy(),
                'boxes': tile_pred['boxes'].numpy(), 
                'labels': tile_pred['labels'].numpy(),
                'scores': tile_pred['scores'].numpy()
            }
            pickle_data['tile_predictions'].append(tile_data)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
    
    def process_time_indices(self, file_path: str, variable_name: str, time_indices: Union[int, List[int]], 
                           use_tiling: bool = False, n_rows: int = 3, n_cols: int = 3, 
                           save_dir: Optional[str] = None, save_pickles: bool = False) -> dict:
        """Process one or more time indices."""
        if isinstance(time_indices, int):
            time_indices = [time_indices]
        
        results = {}
        
        if save_pickles and save_dir:
            pickle_dir = os.path.join(save_dir, "pickles")
            os.makedirs(pickle_dir, exist_ok=True)
        
        for time_idx in time_indices:
            print(f"\nProcessing time index {time_idx}...")
            
            # Load and preprocess
            img_tensor = self.load_and_preprocess(file_path, variable_name, time_idx)
            
            if use_tiling:
                # Tiled prediction
                tile_preds, tile_coords = self.tile_and_predict(img_tensor, n_rows, n_cols)
                
                # Count total instances
                total_instances = sum(pred['masks'].shape[0] for pred in tile_preds)
                print(f"Found {total_instances} total instances across {len(tile_preds)} tiles")
                
                # Stitch results
                canvas_shape = img_tensor.squeeze(0).shape
                stitched_canvas = self.stitch_tiled_predictions(tile_preds, tile_coords, canvas_shape)
                
                # Save pickle if requested
                if save_pickles and save_dir:
                    pickle_path = os.path.join(pickle_dir, f"time_{time_idx}_tiled_masks.pkl")
                    self._save_masks_to_pickle(tile_preds, tile_coords, pickle_path)
                    print(f"Saved masks: {pickle_path}")
                
                # Visualize
                plt.figure(figsize=(12, 12))
                plt.imshow(stitched_canvas, origin="lower")
                plt.title(f"Time {time_idx}: Stitched Predictions ({total_instances} instances)")
                plt.axis('off')
                
                if save_dir:
                    save_path = os.path.join(save_dir, f"time_{time_idx}_tiled_segmentation.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    print(f"Saved: {save_path}")
                
                plt.show()
                
                results[time_idx] = {
                    'tile_predictions': tile_preds,
                    'tile_coords': tile_coords,
                    'total_instances': total_instances,
                    'stitched_canvas': stitched_canvas
                }
            
            else:
                # Full image prediction
                prediction = self.predict_single(img_tensor)
                n_instances = prediction['masks'].shape[0]
                print(f"Found {n_instances} instances")
                
                # Save pickle if requested
                if save_pickles and save_dir:
                    pickle_path = os.path.join(pickle_dir, f"time_{time_idx}_masks.pkl")
                    self._save_prediction_to_pickle(prediction, pickle_path)
                    print(f"Saved masks: {pickle_path}")
                
                # Visualize
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f"time_{time_idx}_segmentation.png")
                
                self.visualize_predictions(prediction, f"Time {time_idx}: Predictions", save_path=save_path)
                
                results[time_idx] = {
                    'prediction': prediction,
                    'n_instances': n_instances
                }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Universal radar segmentation script")
    parser.add_argument("--file_path", required=True, help="Path to NetCDF file")
    parser.add_argument("--variable_name", default="equivalent_reflectivity_factor", 
                       help="NetCDF variable name")
    parser.add_argument("--time_indices", nargs="+", type=int, required=True,
                       help="Time indices to process (space-separated)")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--use_tiling", action="store_true", help="Use tiled prediction")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of tile rows")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of tile columns")
    parser.add_argument("--save_dir", help="Directory to save visualizations")
    parser.add_argument("--save_pickles", action="store_true", 
                       help="Save mask predictions as pickle files")
    
    args = parser.parse_args()
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize segmenter
    segmenter = RadarSegmenter(args.model_path)
    
    # Process time indices
    results = segmenter.process_time_indices(
        file_path=args.file_path,
        variable_name=args.variable_name,
        time_indices=args.time_indices,
        use_tiling=args.use_tiling,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        save_dir=args.save_dir
    )
    
    print(f"\nProcessed {len(results)} time indices successfully!")


if __name__ == "__main__":
    main()