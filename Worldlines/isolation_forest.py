import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import warnings


class StormAnomalyDetector:
    def __init__(self, contamination=0.05, n_estimators=200, max_samples=256):
        """
        Isolation Forest for storm worldline anomaly detection
        
        Args:
            contamination: Expected proportion of anomalies (0.05 = 5%)
            n_estimators: Number of isolation trees
            max_samples: Max samples per tree (256 is sklearn default)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        self.isolation_forest = None
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.raw_data = None
        self.processed_data = None
        self.anomaly_scores = None
        self.anomaly_labels = None
        
    def load_storm_data(self, tensor_path):
        """Load and preprocess storm worldline tensor"""
        print("🌪️  Loading storm data...")
        
        # Load tensor
        data = torch.load(tensor_path, map_location='cpu')
        self.feature_names = data.get('feature_names', [f'param_{i}' for i in range(17)])
        worldlines = data['worldlines']  # [1251, 1878, 17]
        existence_mask = data['existence_mask']  # [1251, 1878]
        
        print(f"📊 Shape: {worldlines.shape}")
        print(f"🏷️  Features: {self.feature_names}")
        
        # Handle missing data using existence mask
        valid_data = []
        storm_ids = []
        timestep_ids = []
        
        for storm_idx in range(worldlines.shape[0]):
            for time_idx in range(worldlines.shape[1]):
                if existence_mask[storm_idx, time_idx]:  # Only include existing data points
                    data_point = worldlines[storm_idx, time_idx].numpy()
                    
                    # Skip if contains NaN or Inf
                    if not (np.isnan(data_point).any() or np.isinf(data_point).any()):
                        valid_data.append(data_point)
                        storm_ids.append(storm_idx)
                        timestep_ids.append(time_idx)
        
        self.raw_data = np.array(valid_data)
        self.storm_ids = np.array(storm_ids)
        self.timestep_ids = np.array(timestep_ids)
        
        print(f"✓ Extracted {len(valid_data):,} valid data points from {worldlines.shape[0]} storms")
        print(f"📈 Data range: [{self.raw_data.min():.2f}, {self.raw_data.max():.2f}]")
        
        return self
    
    def preprocess_data(self, method='robust', apply_pca=False, pca_components=None):
        """
        Preprocess storm data for anomaly detection
        
        Args:
            method: 'standard', 'robust', or 'none'
            apply_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components (None = auto)
        """
        print(f"🔧 Preprocessing with {method} scaling...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            # Robust scaler handles outliers better - good for storm data
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            self.processed_data = self.scaler.fit_transform(self.raw_data)
        else:
            self.processed_data = self.raw_data.copy()
        
        if apply_pca:
            if pca_components is None:
                # Keep 95% of variance
                pca_components = min(15, self.processed_data.shape[1])
            
            print(f"🎯 Applying PCA: {self.processed_data.shape[1]} → {pca_components} dimensions")
            self.pca = PCA(n_components=pca_components)
            self.processed_data = self.pca.fit_transform(self.processed_data)
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"📊 PCA explains {explained_var:.1%} of variance")
        
        print(f"✓ Preprocessed data shape: {self.processed_data.shape}")
        return self
    
    def fit_isolation_forest(self, verbose=True):
        """Fit Isolation Forest model"""
        print("🌳 Training Isolation Forest...")
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Fit the model
        self.isolation_forest.fit(self.processed_data)
        
        # Get anomaly scores and labels
        self.anomaly_scores = self.isolation_forest.decision_function(self.processed_data)
        self.anomaly_labels = self.isolation_forest.predict(self.processed_data)
        
        # Count anomalies
        n_anomalies = np.sum(self.anomaly_labels == -1)
        anomaly_rate = n_anomalies / len(self.anomaly_labels)
        
        if verbose:
            print(f"✓ Model trained on {len(self.processed_data):,} data points")
            print(f"🚨 Found {n_anomalies:,} anomalies ({anomaly_rate:.1%})")
            print(f"📊 Anomaly score range: [{self.anomaly_scores.min():.3f}, {self.anomaly_scores.max():.3f}]")
        
        return self
    
    def get_anomaly_summary(self, top_n=10):
        """Get summary of detected anomalies"""
        anomaly_mask = self.anomaly_labels == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Sort by anomaly score (most anomalous first)
        sorted_indices = anomaly_indices[np.argsort(self.anomaly_scores[anomaly_indices])]
        
        results = []
        for idx in sorted_indices[:top_n]:
            result = {
                'index': idx,
                'storm_id': self.storm_ids[idx],
                'timestep': self.timestep_ids[idx],
                'anomaly_score': self.anomaly_scores[idx],
                'raw_values': self.raw_data[idx].tolist()
            }
            results.append(result)
        
        return results
    
    def plot_anomaly_analysis(self, figsize=(15, 10)):
        """Comprehensive anomaly visualization"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Storm Anomaly Detection Analysis', fontsize=16)
        
        # 1. Anomaly Score Distribution
        axes[0,0].hist(self.anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(self.isolation_forest.offset_, color='red', linestyle='--', 
                         label=f'Threshold: {self.isolation_forest.offset_:.3f}')
        axes[0,0].set_xlabel('Anomaly Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Anomaly Score Distribution')
        axes[0,0].legend()
        
        # 2. Anomalies by Storm
        storm_anomaly_counts = pd.Series(self.storm_ids[self.anomaly_labels == -1]).value_counts()
        top_storms = storm_anomaly_counts.head(20)
        axes[0,1].bar(range(len(top_storms)), top_storms.values)
        axes[0,1].set_xlabel('Storm Rank')
        axes[0,1].set_ylabel('Number of Anomalies')
        axes[0,1].set_title('Top 20 Storms by Anomaly Count')
        
        # 3. Anomalies by Timestep
        timestep_anomaly_counts = pd.Series(self.timestep_ids[self.anomaly_labels == -1]).value_counts()
        axes[0,2].plot(timestep_anomaly_counts.index, timestep_anomaly_counts.values, 'o-', markersize=2)
        axes[0,2].set_xlabel('Timestep')
        axes[0,2].set_ylabel('Number of Anomalies')
        axes[0,2].set_title('Anomalies Across Time')
        
        # 4. PCA Visualization (if PCA was applied)
        if self.pca is not None and self.processed_data.shape[1] >= 2:
            normal_mask = self.anomaly_labels == 1
            anomaly_mask = self.anomaly_labels == -1
            
            axes[1,0].scatter(self.processed_data[normal_mask, 0], 
                            self.processed_data[normal_mask, 1],
                            c='blue', alpha=0.5, s=1, label='Normal')
            axes[1,0].scatter(self.processed_data[anomaly_mask, 0], 
                            self.processed_data[anomaly_mask, 1],
                            c='red', alpha=0.8, s=3, label='Anomaly')
            axes[1,0].set_xlabel('PC1')
            axes[1,0].set_ylabel('PC2')
            axes[1,0].set_title('PCA: Anomalies in Reduced Space')
            axes[1,0].legend()
        else:
            # Feature correlation with anomaly scores
            correlations = []
            for i in range(min(10, self.raw_data.shape[1])):
                corr = np.corrcoef(self.raw_data[:, i], self.anomaly_scores)[0, 1]
                correlations.append(abs(corr))
            
            feature_names = self.feature_names[:len(correlations)] if self.feature_names else [f'F{i}' for i in range(len(correlations))]
            axes[1,0].barh(range(len(correlations)), correlations)
            axes[1,0].set_yticks(range(len(correlations)))
            axes[1,0].set_yticklabels(feature_names, fontsize=8)
            axes[1,0].set_xlabel('|Correlation| with Anomaly Score')
            axes[1,0].set_title('Feature Importance for Anomalies')
        
        # 5. Temporal Pattern of Anomalies
        time_series = np.zeros(1878)  # Max timesteps
        for timestep in self.timestep_ids[self.anomaly_labels == -1]:
            time_series[timestep] += 1
        
        axes[1,1].plot(time_series)
        axes[1,1].set_xlabel('Timestep')
        axes[1,1].set_ylabel('Anomaly Count')
        axes[1,1].set_title('Anomaly Temporal Distribution')
        
        # 6. Most Anomalous Storms Timeline
        top_anomalous_storms = storm_anomaly_counts.head(5).index
        for i, storm_id in enumerate(top_anomalous_storms):
            storm_mask = (self.storm_ids == storm_id) & (self.anomaly_labels == -1)
            storm_timesteps = self.timestep_ids[storm_mask]
            y_pos = [i] * len(storm_timesteps)
            axes[1,2].scatter(storm_timesteps, y_pos, alpha=0.7, s=20)
        
        axes[1,2].set_xlabel('Timestep')
        axes[1,2].set_ylabel('Storm ID Rank')
        axes[1,2].set_title('Anomaly Timeline for Top 5 Storms')
        axes[1,2].set_yticks(range(5))
        axes[1,2].set_yticklabels([f'Storm {sid}' for sid in top_anomalous_storms])
        
        plt.tight_layout()
        return fig
    
    def save_anomaly_results(self, filename='storm_anomalies.csv'):
        """Save anomaly detection results to CSV"""
        results_df = pd.DataFrame({
            'index': range(len(self.anomaly_scores)),
            'storm_id': self.storm_ids,
            'timestep': self.timestep_ids,
            'anomaly_score': self.anomaly_scores,
            'is_anomaly': self.anomaly_labels == -1
        })
        
        # Add raw feature values
        for i, feature_name in enumerate(self.feature_names or [f'feature_{i}' for i in range(17)]):
            results_df[feature_name] = self.raw_data[:, i]
        
        results_df.to_csv(filename, index=False)
        print(f"💾 Saved results to {filename}")
        
        return results_df

# Usage Example
def main():
    # Initialize detector
    detector = StormAnomalyDetector(
        contamination=0.05,  # Expect 5% anomalies
        n_estimators=200,
        max_samples=256
    )
    
    # Load and process data
    detector.load_storm_data('/home/mwigder/WL-CTN/prelims/worldline_tensor.pt')
    detector.preprocess_data(method='robust', apply_pca=False)  # Try with PCA later
    
    # Train isolation forest
    detector.fit_isolation_forest()
    
    # Get top anomalies
    print("\n🚨 TOP 10 MOST ANOMALOUS STORM EVENTS:")
    print("=" * 60)
    
    top_anomalies = detector.get_anomaly_summary(top_n=10)
    for i, anomaly in enumerate(top_anomalies, 1):
        print(f"{i:2d}. Storm {anomaly['storm_id']:4d}, Timestep {anomaly['timestep']:4d} "
              f"(Score: {anomaly['anomaly_score']:6.3f})")
    
    # Create visualizations
    fig = detector.plot_anomaly_analysis()
    plt.savefig('storm_anomaly_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved analysis plots to 'storm_anomaly_analysis.png'")
    
    # Save results
    results_df = detector.save_anomaly_results()
    
    print(f"\n✅ Analysis complete! Found {np.sum(detector.anomaly_labels == -1):,} anomalous storm events")
    
    return detector, results_df

if __name__ == "__main__":
    detector, results = main()