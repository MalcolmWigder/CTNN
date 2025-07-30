#!/usr/bin/env python3
"""
Storm Analysis Toolkit - Simple Main Interface
"""

from segment import RadarSegmenter
from train_new_model import train_new_model
from pack_tensor_data import TensorPacker
from isolation_forest import StormAnomalyDetector
import subprocess
import sys


def pack_data(radar_dir, mask_dir, output_file="training_data.pt"):
    """Pack radar and mask files into training tensor"""
    packer = TensorPacker()
    packer.pack_simple_directories(radar_dir, mask_dir, output_file)
    print(f"Data packed to {output_file}")
    return output_file


def train_model(data_file, output_dir="./model_output", epochs=20, quick=False):
    """Train a new segmentation model"""
    if quick:
        train_new_model(
            tensor_path=data_file,
            output_dir=output_dir,
            train_samples=100,
            val_samples=50,
            epochs=5
        )
    else:
        train_new_model(
            tensor_path=data_file,
            output_dir=output_dir,
            epochs=epochs
        )
    
    print(f"Model saved to {output_dir}")
    return f"{output_dir}/best_model.pth"


def segment_data(nc_file, model_path, time_indices=None, save_dir="./results"):
    """Segment radar data"""
    if time_indices is None:
        time_indices = [0]
    
    segmenter = RadarSegmenter(model_path)
    results = segmenter.process_time_indices(
        file_path=nc_file,
        variable_name="equivalent_reflectivity_factor",
        time_indices=time_indices,
        save_dir=save_dir,
        save_pickles=True
    )
    
    print(f"Segmentation results saved to {save_dir}")
    return results


def analyze_worldlines(worldline_tensor):
    """Run worldlines analysis"""
    print("Running storm evolution analysis...")
    subprocess.run([sys.executable, "summarize_WL.py"])
    
    print("Running anomaly detection...")
    detector = StormAnomalyDetector()
    detector.load_storm_data(worldline_tensor)
    detector.preprocess_data()
    detector.fit_isolation_forest()
    
    detector.plot_anomaly_analysis()
    detector.save_anomaly_results()
    
    print("Worldlines analysis complete")


# Example usage functions
def full_segmentation_pipeline(radar_dir, mask_dir, nc_file):
    """Complete segmentation workflow"""
    # Step 1: Pack data
    data_file = pack_data(radar_dir, mask_dir)
    
    # Step 2: Train model
    model_path = train_model(data_file)
    
    # Step 3: Segment new data
    results = segment_data(nc_file, model_path, time_indices=[0, 1, 2])
    
    return results


def quick_segment(nc_file, model_path, time_indices):
    """Quick segmentation with existing model"""
    return segment_data(nc_file, model_path, time_indices)


if __name__ == "__main__":
    # Example usage
    print("Storm Analysis Toolkit")
    print("Import this file and use the functions directly:")
    print("")
    print("# Pack training data")
    print("pack_data('/path/to/radar', '/path/to/masks')")
    print("")
    print("# Train model")
    print("train_model('training_data.pt')")
    print("")
    print("# Segment data")
    print("segment_data('data.nc', 'model_output/best_model.pth', [0,1,2])")
    print("")
    print("# Analyze worldlines")
    print("analyze_worldlines('worldline_tensor.pt')")