# Storm Analysis Toolkit

A comprehensive Python package for radar-based storm segmentation and temporal analysis using deep learning and anomaly detection.

## Overview

This toolkit provides two main analysis pipelines:
1. **Segmentation Pipeline**: Deep learning-based storm detection and segmentation from radar data
2. **Worldlines Pipeline**: Temporal storm tracking, evolution analysis, and anomaly detection

## Installation

```bash
git clone <repository-url>
cd storm-analysis-toolkit
pip install -r requirements.txt
```

### Requirements
- **GPU**: NVIDIA GPU with 10GB+ VRAM recommended
- **Python**: 3.8+
- **Key Dependencies**: PyTorch, torchvision, netCDF4, scikit-learn, matplotlib

## Getting Started

### 1. Clone and Install
```bash
git clone https://github.com/MalcolmWigder/CTNN.git
cd CTNN
pip install -r requirements.txt
```

### 2. Download Data and Models
**Download from Google Drive**: [Large Files Folder](https://drive.google.com/drive/folders/11pDrtQ_ldfBaaSEP2szd1x6fRfBk98kR?usp=sharing)

Download these files and place them in the correct locations:
- `best_model.pth` → `Segmentation/best_model.pth`
- `worldline_tensor.pt` → `Worldlines/worldline_tensor.pt`
- `tensorstore.pt` (training data) → `Segmentation/tensorstore.pt` (note that this file is 26 GB)

### 3. Quick Test
```python
from main import segment_data, analyze_worldlines

# Test segmentation (you'll need a .nc file)
results = segment_data("your_radar_data.nc", "Segmentation/best_model.pth", [0, 1, 2])

# Test worldlines analysis
analyze_worldlines("Worldlines/worldline_tensor.pt")
```

### Command Line Usage
```bash
# Pack training data
python pack_tensor_data.py --simple_dirs /radar/data /mask/data --output training.pt

# Train new model  
python train_new_model.py --tensor_path training.pt --epochs 30

# Segment radar data
python segment.py --file_path data.nc --time_indices 0 1 2 --model_path best_model.pth --save_pickles

# Analyze worldlines
python summarize_WL.py  # storm evolution and movement analysis
python isolation_forest.py  # anomaly detection
```

## Workflows

### 1. Segmentation Workflow
For detecting and segmenting storm cells in radar data:

```
Radar Data (.nc) → [Segmentation Model] → Instance Masks → Visualizations + Pickles
```

**Steps:**
1. **Data Preparation**: Use `pack_tensor_data.py` to prepare training data
2. **Model Training**: Use `train_new_model.py` to train segmentation model  
3. **Segmentation**: Use `segment.py` to segment new radar data
4. **Evaluation**: Use `evaluate_model.py` to assess model performance

### 2. Worldlines Workflow
For temporal storm tracking and anomaly detection:

```  
Worldline Tensor → [Analysis Tools] → Evolution Patterns + Anomaly Detection
```

**Steps:**
1. **Database Creation**: Use `create_WL_database.py` to build worldline database
2. **Evolution Analysis**: Use `summarize_WL.py` for movement and morphology patterns
3. **Anomaly Detection**: Use `isolation_forest.py` to find unusual storm behavior

## Core Components

### Segmentation Module
- **`segment.py`**: Universal segmentation script with tiling support
- **`train_new_model.py`**: Simplified model training interface
- **`segmentation_model_v4.py`**: Core Mask R-CNN implementation  
- **`evaluate_model.py`**: Model performance evaluation
- **`pack_tensor_data.py`**: Flexible data packing utility

### Worldlines Module  
- **`create_WL_database.py`**: Build temporal storm tracking database
- **`summarize_WL.py`**: Storm evolution and movement analysis
- **`isolation_forest.py`**: Anomaly detection in storm behavior
- **`3D_vis.py`**: 3D visualization tools

### Utilities
- **`tensor_packer_helper.py`**: Data format conversion helpers
- **`training_data_vis.py`**: Training data visualization tools

## Data Formats

### Input Data
- **Radar Files**: NetCDF (.nc) format with radar reflectivity data
- **Training Masks**: Instance segmentation masks (numpy arrays)
- **Worldline Tensors**: Preprocessed storm tracking data (.pt format)

### Output Data
- **Segmentation Results**: PNG visualizations + pickle files with masks
- **Analysis Plots**: Storm evolution patterns, movement analysis, anomalies
- **Data Files**: CSV results, trained models (.pth), worldline databases

## Key Features

### Segmentation
- **Multi-scale Processing**: Tiling support for large radar images
- **Flexible Input**: Handles various radar data formats and time indices
- **Rich Output**: Visualizations, pickle exports, and performance metrics
- **GPU Acceleration**: CUDA support with mixed precision training

### Worldlines Analysis
- **Evolution Tracking**: Analyze how storms change over time
- **Movement Patterns**: Direction and speed analysis with polar plots
- **Anomaly Detection**: Isolation Forest-based unusual behavior detection
- **Comprehensive Visualization**: Evolution curves, movement vectors, temporal patterns

## Examples

<details>
<summary><b>Example 1: Train and Use Segmentation Model</b></summary>

```python
# 1. Prepare training data
from pack_tensor_data import TensorPacker

packer = TensorPacker()
packer.pack_simple_directories(
    radar_dir="/path/to/radar",
    mask_dir="/path/to/masks", 
    out_path="training_data.pt"
)

# 2. Train model
from train_new_model import train_new_model

train_new_model(
    tensor_path="training_data.pt",
    output_dir="./my_model",
    epochs=25,
    train_samples=800
)

# 3. Segment new data
from segment import RadarSegmenter

segmenter = RadarSegmenter("./my_model/best_model.pth")
results = segmenter.process_time_indices(
    file_path="new_radar_data.nc",
    variable_name="equivalent_reflectivity_factor",
    time_indices=[0, 5, 10],
    save_dir="./results",
    save_pickles=True
)
```
</details>

<details>
<summary><b>Example 2: Worldlines Analysis</b></summary>

```python
# 1. Analyze storm evolution
import subprocess
subprocess.run(["python", "summarize_WL.py"])

# 2. Detect anomalies
from isolation_forest import StormAnomalyDetector

detector = StormAnomalyDetector(contamination=0.05)
detector.load_storm_data("worldline_tensor.pt")
detector.preprocess_data(method='robust')
detector.fit_isolation_forest()

# Get top anomalies
anomalies = detector.get_anomaly_summary(top_n=20)
detector.plot_anomaly_analysis()
detector.save_anomaly_results("storm_anomalies.csv")
```
</details>

<details>
<summary><b>Example 3: Custom Analysis Pipeline</b></summary>

```python
from main import StormAnalysisToolkit

# Custom workflow
toolkit = StormAnalysisToolkit()

# Train with custom parameters
toolkit.train_segmentation_model(
    training_data="my_data.pt",
    model_name="storm_detector_v2",
    quick_train=False,
    epochs=40
)

# Segment multiple files
radar_files = ["file1.nc", "file2.nc", "file3.nc"]
for file in radar_files:
    toolkit.segment_radar_data(
        file, 
        time_indices=list(range(0, 50, 5)),  # Every 5th frame
        save_pickles=True
    )

# Analyze results
toolkit.analyze_worldlines("worldline_tensor.pt", detect_anomalies=True)
```
</details>

## Output Examples

### Segmentation Results
- **Visualizations**: Side-by-side original radar + detected storm cells
- **Pickle Files**: Raw mask data for downstream processing
- **Performance Metrics**: IoU, precision, recall, training curves

### Worldlines Analysis
- **Evolution Patterns**: How storm area, shape, intensity change over time
- **Movement Analysis**: Direction histograms, speed distributions, polar plots
- **Anomaly Detection**: Unusual storms with scores and detailed analysis

## Important Notes

### Training Data Requirements
- **Input Images**: Radar reflectivity data, preferably cropped to standard size
- **Ground Truth**: Instance segmentation masks (not semantic masks)
- **Format**: Use `pack_tensor_data.py` to ensure proper formatting

### Hardware Requirements  
- **GPU Memory**: 10GB+ VRAM recommended for training
- **CPU**: Multi-core processor for data loading
- **Storage**: Several GB for model checkpoints and analysis outputs

### Performance Tips
- **Tiling**: Use `--use_tiling` for large radar images (>1000x1000 pixels)
- **Batch Size**: Keep at 1 for training due to memory constraints
- **Mixed Precision**: Enabled by default for memory efficiency

## Troubleshooting

<details>
<summary><b>Common Issues</b></summary>

**Out of Memory Errors**
- Reduce batch size or use tiling
- Enable mixed precision training
- Clear GPU cache between runs

**Data Loading Issues**  
- Check file paths and permissions
- Verify NetCDF variable names
- Ensure training data format with `pack_tensor_data.py`

**Poor Segmentation Results**
- Increase training epochs
- Check training data quality
- Adjust anchor sizes for storm size
- Use more training samples

**Worldlines Analysis Errors**
- Verify worldline tensor format
- Check for NaN values in data
- Ensure sufficient storm tracks for analysis
</details>

## API Reference

### Main Classes
- **`RadarSegmenter`**: Core segmentation functionality
- **`StormAnomalyDetector`**: Isolation Forest anomaly detection  
- **`TensorPacker`**: Data preparation utilities
- **`StormAnalysisToolkit`**: High-level interface

### Key Functions
- **`process_time_indices()`**: Segment multiple time steps
- **`train_new_model()`**: Train segmentation model
- **`pack_tensor_data()`**: Prepare training data
- **`load_storm_data()`**: Load worldline tensors
- **`fit_isolation_forest()`**: Train anomaly detector

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Made possible by the SULI program at Brookhaven National Laboratory
- Thank you to my mentor, Die Wang
- Thank you to the team who built CoCoMET: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1328/
- Built on PyTorch and Mask R-CNN architecture
- Utilizes scikit-learn for anomaly detection
- NetCDF4 for radar data handling
- Matplotlib/Seaborn for visualizations