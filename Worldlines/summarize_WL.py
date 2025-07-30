#!/usr/bin/env python3
"""
Storm Evolution and Direction Analysis from Worldline Tensor
Analyzes storm morphology evolution and movement patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the worldline tensor data"""
    data = torch.load("/home/mwigder/WL-CTN/prelims/worldline_tensor.pt", map_location='cpu')
    
    worldlines = data['worldlines'].numpy()  # [1251, 1878, 17]
    existence_mask = data['existence_mask'].numpy()  # [1251, 1878]
    timestamps = data['timestamps']
    feature_names = data['feature_names']
    
    return worldlines, existence_mask, timestamps, feature_names

def analyze_storm_evolution():
    """Analyze how storms normally evolve over time"""
    print("Analyzing storm evolution patterns...")
    
    worldlines, existence_mask, timestamps, feature_names = load_data()
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('How Storms Normally Evolve', fontsize=16, fontweight='bold')
    
    # Key evolution metrics
    key_features = {
        'Area': 2,
        'Perimeter': 3,
        'Major Axis': 7,
        'Minor Axis': 8,
        'Compactness': 11,
        'Velocity Magnitude': None,  # Will calculate
        'Acceleration Magnitude': None,  # Will calculate
        'Area Change Rate': 16,
        'Aspect Ratio': 6
    }
    
    # Calculate velocity and acceleration magnitudes
    velocity_mag = np.sqrt(worldlines[:, :, 12]**2 + worldlines[:, :, 13]**2)
    accel_mag = np.sqrt(worldlines[:, :, 14]**2 + worldlines[:, :, 15]**2)
    
    plot_idx = 0
    
    for feature_name, feature_idx in key_features.items():
        if plot_idx >= 9:
            break
            
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        if feature_name == 'Velocity Magnitude':
            feature_data = velocity_mag
        elif feature_name == 'Acceleration Magnitude':
            feature_data = accel_mag
        else:
            feature_data = worldlines[:, :, feature_idx]
        
        # Get evolution patterns for existing storms
        evolution_curves = []
        for storm_id in range(worldlines.shape[0]):
            storm_exists = existence_mask[storm_id, :]
            if storm_exists.sum() > 5:  # At least 5 timesteps
                storm_feature = feature_data[storm_id, storm_exists]
                if len(storm_feature) > 5:
                    # Normalize to storm length for comparison
                    norm_time = np.linspace(0, 1, len(storm_feature))
                    evolution_curves.append((norm_time, storm_feature))
        
        if evolution_curves:
            # Plot individual storm curves (sample)
            for i, (norm_time, values) in enumerate(evolution_curves[:20]):
                ax.plot(norm_time, values, alpha=0.3, color='lightblue', linewidth=0.5)
            
            # Calculate average evolution
            # Interpolate all curves to common grid
            common_grid = np.linspace(0, 1, 50)
            interpolated_curves = []
            for norm_time, values in evolution_curves:
                if len(values) > 1:
                    interp_values = np.interp(common_grid, norm_time, values)
                    interpolated_curves.append(interp_values)
            
            if interpolated_curves:
                interpolated_curves = np.array(interpolated_curves)
                mean_evolution = np.mean(interpolated_curves, axis=0)
                std_evolution = np.std(interpolated_curves, axis=0)
                
                ax.plot(common_grid, mean_evolution, 'red', linewidth=3, label='Average')
                ax.fill_between(common_grid, mean_evolution - std_evolution, 
                               mean_evolution + std_evolution, alpha=0.3, color='red')
        
        ax.set_title(f'{feature_name} Evolution')
        ax.set_xlabel('Normalized Storm Lifetime')
        ax.set_ylabel(feature_name)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('storm_evolution_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Storm evolution analysis saved as 'storm_evolution_patterns.png'")

def analyze_storm_directions():
    """Analyze storm movement directions and patterns"""
    print("Analyzing storm movement directions...")
    
    worldlines, existence_mask, timestamps, feature_names = load_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Storm Movement Direction Analysis', fontsize=16, fontweight='bold')
    
    # Extract movement data from all storms
    all_velocities_x = []
    all_velocities_y = []
    all_speeds = []
    all_directions = []
    storm_tracks = []
    
    for storm_id in range(worldlines.shape[0]):
        storm_exists = existence_mask[storm_id, :]
        if storm_exists.sum() > 2:  # Need at least 3 points for meaningful track
            # Get storm positions and velocities
            x_pos = worldlines[storm_id, storm_exists, 0]  # centroid_x
            y_pos = worldlines[storm_id, storm_exists, 1]  # centroid_y
            vel_x = worldlines[storm_id, storm_exists, 12]
            vel_y = worldlines[storm_id, storm_exists, 13]
            
            # Store track
            storm_tracks.append((x_pos, y_pos))
            
            # Calculate speeds and directions
            speeds = np.sqrt(vel_x**2 + vel_y**2)
            directions = np.arctan2(vel_y, vel_x) * 180 / np.pi
            directions = (directions + 360) % 360  # Normalize to 0-360
            
            all_velocities_x.extend(vel_x)
            all_velocities_y.extend(vel_y)
            all_speeds.extend(speeds)
            all_directions.extend(directions)
    
    all_speeds = np.array(all_speeds)
    all_directions = np.array(all_directions)
    
    # Remove zero velocities for cleaner analysis
    nonzero_mask = all_speeds > 0.1
    clean_speeds = all_speeds[nonzero_mask]
    clean_directions = all_directions[nonzero_mask]
    
    # Plot 1: Direction histogram
    ax = axes[0, 0]
    ax.hist(clean_directions, bins=36, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Storm Movement Direction Distribution')
    ax.set_xlabel('Direction (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 360)
    
    # Add cardinal directions
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        ax.axvline(angle, color='red', linestyle='--', alpha=0.5)
        ax.text(angle, ax.get_ylim()[1]*0.9, label, ha='center', color='red', fontweight='bold')
    
    # Plot 2: Speed distribution
    ax = axes[0, 1]
    ax.hist(clean_speeds, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_title('Storm Movement Speed Distribution')
    ax.set_xlabel('Speed (pixels/timestep)')
    ax.set_ylabel('Frequency')
    
    # Plot 3: Polar plot of directions
    ax = plt.subplot(2, 3, 3, projection='polar')
    theta = clean_directions * np.pi / 180
    ax.hist(theta, bins=36, alpha=0.7, color='lightgreen')
    ax.set_title('Movement Directions (Polar)')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_zero_location('N')  # North at top
    
    # Plot 4: Velocity vector field
    ax = axes[1, 0]
    vel_x_clean = np.array(all_velocities_x)[nonzero_mask]
    vel_y_clean = np.array(all_velocities_y)[nonzero_mask]
    
    # Sample for visualization
    sample_size = min(1000, len(vel_x_clean))
    sample_idx = np.random.choice(len(vel_x_clean), sample_size, replace=False)
    
    ax.scatter(vel_x_clean[sample_idx], vel_y_clean[sample_idx], 
               alpha=0.5, s=20, c=clean_speeds[sample_idx], cmap='viridis')
    ax.set_title('Velocity Vector Distribution')
    ax.set_xlabel('Velocity X')
    ax.set_ylabel('Velocity Y')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 5: Speed vs Direction
    ax = axes[1, 1]
    ax.scatter(clean_directions, clean_speeds, alpha=0.5, s=10)
    ax.set_title('Speed vs Direction')
    ax.set_xlabel('Direction (degrees)')
    ax.set_ylabel('Speed (pixels/timestep)')
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Sample storm tracks
    ax = axes[1, 2]
    for i, (x_pos, y_pos) in enumerate(storm_tracks[:30]):  # Show first 30 tracks
        ax.plot(x_pos, y_pos, alpha=0.6, linewidth=1)
        ax.scatter(x_pos[0], y_pos[0], color='green', s=20, alpha=0.7)  # Start
        ax.scatter(x_pos[-1], y_pos[-1], color='red', s=20, alpha=0.7)  # End
    
    ax.set_title('Sample Storm Tracks\n(Green=Start, Red=End)')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('storm_movement_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Storm movement analysis saved as 'storm_movement_analysis.png'")
    
    # Print summary statistics
    print(f"\nMOVEMENT SUMMARY:")
    print(f"Total movement observations: {len(clean_speeds):,}")
    print(f"Mean speed: {np.mean(clean_speeds):.2f} pixels/timestep")
    print(f"Speed range: {np.min(clean_speeds):.2f} - {np.max(clean_speeds):.2f}")
    print(f"Most common direction: {stats.mode(np.round(clean_directions/10)*10, keepdims=True)[0][0]:.0f}°")
    print(f"Direction std: {np.std(clean_directions):.1f}°")

def generate_summary_plots():
    """Generate additional summary visualizations"""
    print("Generating summary plots...")
    
    worldlines, existence_mask, timestamps, feature_names = load_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Storm Dataset Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Storm lifetimes
    ax = axes[0, 0]
    lifetimes = existence_mask.sum(axis=1)
    lifetimes = lifetimes[lifetimes > 0]  # Remove storms with 0 lifetime
    
    ax.hist(lifetimes, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title('Storm Lifetime Distribution')
    ax.set_xlabel('Lifetime (timesteps)')
    ax.set_ylabel('Number of Storms')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Storms active over time
    ax = axes[0, 1]
    active_storms = existence_mask.sum(axis=0)
    time_indices = range(len(active_storms))
    
    ax.plot(time_indices, active_storms, linewidth=1, alpha=0.8)
    ax.set_title('Active Storms Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Active Storms')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Area distribution
    ax = axes[1, 0]
    all_areas = worldlines[:, :, 2][existence_mask]
    all_areas = all_areas[all_areas > 0]
    
    ax.hist(all_areas, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title('Storm Area Distribution')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Compactness vs Area
    ax = axes[1, 1]
    all_compactness = worldlines[:, :, 11][existence_mask]
    area_sample = all_areas[:len(all_compactness)]
    
    # Sample for visualization
    sample_size = min(5000, len(all_compactness))
    sample_idx = np.random.choice(len(all_compactness), sample_size, replace=False)
    
    ax.scatter(area_sample[sample_idx], all_compactness[sample_idx], 
               alpha=0.5, s=10, color='teal')
    ax.set_title('Storm Compactness vs Area')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Compactness')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('storm_dataset_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Dataset summary saved as 'storm_dataset_summary.png'")
    
    # Print dataset statistics
    print(f"\nDATASET SUMMARY:")
    print(f"Total storms: {worldlines.shape[0]:,}")
    print(f"Total timesteps: {worldlines.shape[1]:,}")
    print(f"Time span: {timestamps[0]} to {timestamps[-1]}")
    print(f"Mean storm lifetime: {np.mean(lifetimes):.1f} timesteps")
    print(f"Max concurrent storms: {active_storms.max()}")
    print(f"Mean area: {np.mean(all_areas):.1f} pixels²")

if __name__ == "__main__":
    print("Starting Storm Worldline Analysis...")
    print("="*50)
    
    analyze_storm_evolution()
    analyze_storm_directions() 
    generate_summary_plots()
    
    print("\n" + "="*50)
    print("Analysis complete! Generated files:")
    print("• storm_evolution_patterns.png")
    print("• storm_movement_analysis.png") 
    print("• storm_dataset_summary.png")