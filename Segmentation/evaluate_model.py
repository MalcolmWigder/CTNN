import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import json
import os

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union for binary masks"""
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def match_predictions_to_ground_truth(pred_boxes, pred_masks, gt_boxes, gt_masks, iou_threshold=0.5):
    """Match predictions to ground truth using Hungarian algorithm"""
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    
    if num_pred == 0 or num_gt == 0:
        return [], [], list(range(num_gt))
    
    # Calculate IoU matrix
    iou_matrix = torch.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            # Box IoU
            box1 = pred_boxes[i]
            box2 = gt_boxes[j]
            
            x1 = torch.max(box1[0], box2[0])
            y1 = torch.max(box1[1], box2[1])
            x2 = torch.min(box1[2], box2[2])
            y2 = torch.min(box1[3], box2[3])
            
            intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            box_iou = intersection / union if union > 0 else 0
            
            # Mask IoU (if both have masks)
            if len(pred_masks) > i and len(gt_masks) > j:
                mask_iou = calculate_iou(pred_masks[i] > 0.5, gt_masks[j] > 0)
                iou_matrix[i, j] = (box_iou + mask_iou) / 2
            else:
                iou_matrix[i, j] = box_iou
    
    # Find matches
    matched_pred = []
    matched_gt = []
    
    while iou_matrix.numel() > 0 and iou_matrix.max() > iou_threshold:
        max_iou = iou_matrix.max()
        max_idx = iou_matrix.argmax()
        pred_idx = max_idx // iou_matrix.shape[1]
        gt_idx = max_idx % iou_matrix.shape[1]
        
        matched_pred.append(pred_idx.item())
        matched_gt.append(gt_idx.item())
        
        # Remove matched entries
        iou_matrix[pred_idx, :] = -1
        iou_matrix[:, gt_idx] = -1
    
    # Find unmatched ground truth
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    
    return matched_pred, matched_gt, unmatched_gt

@torch.no_grad()
def evaluate_model(model, dataloader, device, score_threshold=0.5, iou_threshold=0.5):
    """Comprehensive evaluation of the model"""
    model.eval()
    
    # Metrics storage
    metrics = defaultdict(list)
    all_predictions = []
    
    print("Running evaluation...")
    
    for images, targets in tqdm(dataloader):
        if len(images) == 0:
            continue
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Get predictions
        outputs = model(images)
        
        # Process each image
        for img_idx, (output, target) in enumerate(zip(outputs, targets)):
            # Filter predictions by score
            keep = output['scores'] > score_threshold
            pred_boxes = output['boxes'][keep]
            pred_scores = output['scores'][keep]
            pred_masks = output['masks'][keep]
            
            gt_boxes = target['boxes']
            gt_masks = target['masks']
            
            # Match predictions to ground truth
            matched_pred, matched_gt, unmatched_gt = match_predictions_to_ground_truth(
                pred_boxes, pred_masks, gt_boxes, gt_masks, iou_threshold
            )
            
            # Calculate metrics
            num_tp = len(matched_pred)
            num_fp = len(pred_boxes) - num_tp
            num_fn = len(unmatched_gt)
            
            metrics['true_positives'].append(num_tp)
            metrics['false_positives'].append(num_fp)
            metrics['false_negatives'].append(num_fn)
            
            # Calculate IoU for matched pairs
            for pred_idx, gt_idx in zip(matched_pred, matched_gt):
                if pred_idx < len(pred_masks) and gt_idx < len(gt_masks):
                    mask_iou = calculate_iou(
                        pred_masks[pred_idx][0] > 0.5,
                        gt_masks[gt_idx] > 0
                    )
                    metrics['mask_ious'].append(mask_iou.item())
            
            # Store prediction info
            all_predictions.append({
                'num_predictions': len(pred_boxes),
                'num_ground_truth': len(gt_boxes),
                'scores': pred_scores.cpu().tolist(),
                'matched': num_tp,
                'false_positives': num_fp,
                'false_negatives': num_fn
            })
    
    # Calculate aggregate metrics
    total_tp = sum(metrics['true_positives'])
    total_fp = sum(metrics['false_positives'])
    total_fn = sum(metrics['false_negatives'])
    
    # Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average IoU
    avg_iou = np.mean(metrics['mask_ious']) if metrics['mask_ious'] else 0
    
    # Per-image statistics
    avg_pred_per_image = np.mean([p['num_predictions'] for p in all_predictions])
    avg_gt_per_image = np.mean([p['num_ground_truth'] for p in all_predictions])
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'average_mask_iou': avg_iou,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'avg_predictions_per_image': avg_pred_per_image,
        'avg_ground_truth_per_image': avg_gt_per_image,
        'num_images_evaluated': len(all_predictions),
        'score_threshold': score_threshold,
        'iou_threshold': iou_threshold
    }
    
    return results, all_predictions

def plot_evaluation_results(results, predictions, save_path):
    """Create comprehensive evaluation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Precision-Recall summary
    ax = axes[0, 0]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [results['precision'], results['recall'], results['f1_score']]
    colors = ['blue', 'green', 'red']
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Detection Metrics')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Detection counts
    ax = axes[0, 1]
    categories = ['True Positives', 'False Positives', 'False Negatives']
    counts = [results['total_true_positives'], 
              results['total_false_positives'], 
              results['total_false_negatives']]
    colors = ['green', 'orange', 'red']
    ax.bar(categories, counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Detection Counts')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count + max(counts)*0.02, str(count), ha='center', va='bottom')
    
    # 3. IoU distribution
    ax = axes[0, 2]
    all_scores = []
    for pred in predictions:
        all_scores.extend(pred['scores'])
    if all_scores:
        ax.hist(all_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(results['score_threshold'], color='red', linestyle='--', 
                   label=f'Threshold: {results["score_threshold"]}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Confidence Distribution')
        ax.legend()
    
    # 4. Predictions vs Ground Truth
    ax = axes[1, 0]
    pred_counts = [p['num_predictions'] for p in predictions]
    gt_counts = [p['num_ground_truth'] for p in predictions]
    ax.scatter(gt_counts, pred_counts, alpha=0.5)
    max_count = max(max(pred_counts) if pred_counts else 1, 
                    max(gt_counts) if gt_counts else 1) + 1
    ax.plot([0, max_count], [0, max_count], 'r--', label='Perfect prediction')
    ax.set_xlabel('Ground Truth Objects')
    ax.set_ylabel('Predicted Objects')
    ax.set_title('Predictions vs Ground Truth Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Per-image performance
    ax = axes[1, 1]
    tp_per_image = [p['matched'] for p in predictions]
    fp_per_image = [p['false_positives'] for p in predictions]
    fn_per_image = [p['false_negatives'] for p in predictions]
    
    if len(predictions) <= 50:
        x = range(len(predictions))
        width = 0.25
        ax.bar([i - width for i in x], tp_per_image, width, label='TP', color='green', alpha=0.7)
        ax.bar(x, fp_per_image, width, label='FP', color='orange', alpha=0.7)
        ax.bar([i + width for i in x], fn_per_image, width, label='FN', color='red', alpha=0.7)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Count')
        ax.set_title('Per-Image Detection Performance')
        ax.legend()
    else:
        # For many images, show histogram
        ax.hist([tp_per_image, fp_per_image, fn_per_image], 
                bins=10, label=['TP', 'FP', 'FN'], 
                color=['green', 'orange', 'red'], alpha=0.7)
        ax.set_xlabel('Count per Image')
        ax.set_ylabel('Frequency')
        ax.set_title('Detection Performance Distribution')
        ax.legend()
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""Evaluation Summary
    
Total Images: {results['num_images_evaluated']}
Score Threshold: {results['score_threshold']}
IoU Threshold: {results['iou_threshold']}

Performance Metrics:
• Precision: {results['precision']:.3f}
• Recall: {results['recall']:.3f}
• F1-Score: {results['f1_score']:.3f}
• Avg Mask IoU: {results['average_mask_iou']:.3f}

Detection Statistics:
• Avg GT/image: {results['avg_ground_truth_per_image']:.2f}
• Avg Pred/image: {results['avg_predictions_per_image']:.2f}
• Total TP: {results['total_true_positives']}
• Total FP: {results['total_false_positives']}
• Total FN: {results['total_false_negatives']}"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def evaluate_at_multiple_thresholds(model, dataloader, device, score_thresholds=[0.3, 0.5, 0.7]):
    """Evaluate model at multiple score thresholds"""
    results_by_threshold = {}
    
    for threshold in score_thresholds:
        print(f"\nEvaluating at score threshold: {threshold}")
        results, predictions = evaluate_model(
            model, dataloader, device, 
            score_threshold=threshold, 
            iou_threshold=0.5
        )
        results_by_threshold[threshold] = results
    
    # Plot threshold comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    thresholds = list(results_by_threshold.keys())
    precisions = [results_by_threshold[t]['precision'] for t in thresholds]
    recalls = [results_by_threshold[t]['recall'] for t in thresholds]
    f1_scores = [results_by_threshold[t]['f1_score'] for t in thresholds]
    
    # Precision-Recall curve
    ax1.plot(recalls, precisions, 'b-o', linewidth=2, markersize=8)
    for i, t in enumerate(thresholds):
        ax1.annotate(f'{t:.1f}', (recalls[i], precisions[i]), 
                    textcoords="offset points", xytext=(5,5))
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # F1 score vs threshold
    ax2.plot(thresholds, f1_scores, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Score Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Score Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, results_by_threshold

def create_qualitative_results(model, dataset, device, num_samples=10, save_dir='qualitative_results'):
    """Generate qualitative results showing predictions on sample images"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        if sample is None:
            continue
            
        image, target = sample
        
        # Get prediction
        with torch.no_grad():
            prediction = model([image.to(device)])[0]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax = axes[0]
        img_display = image[0].cpu().numpy()
        ax.imshow(img_display, cmap='gray')
        ax.set_title('Input Radar Image')
        ax.axis('off')
        
        # Ground truth
        ax = axes[1]
        ax.imshow(img_display, cmap='gray')
        for box in target['boxes']:
            rect = patches.Rectangle((box[0], box[1]), 
                                   box[2]-box[0], box[3]-box[1],
                                   linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Show GT masks
        if len(target['masks']) > 0:
            combined_mask = torch.zeros_like(target['masks'][0])
            for mask in target['masks']:
                combined_mask = torch.maximum(combined_mask, mask)
            ax.imshow(combined_mask.cpu().numpy(), alpha=0.3, cmap='Greens')
        
        ax.set_title(f'Ground Truth ({len(target["boxes"])} clouds)')
        ax.axis('off')
        
        # Predictions
        ax = axes[2]
        ax.imshow(img_display, cmap='gray')
        
        # Filter by score
        keep = prediction['scores'] > 0.5
        pred_boxes = prediction['boxes'][keep].cpu()
        pred_scores = prediction['scores'][keep].cpu()
        pred_masks = prediction['masks'][keep].cpu()
        
        for box, score in zip(pred_boxes, pred_scores):
            rect = patches.Rectangle((box[0], box[1]), 
                                   box[2]-box[0], box[3]-box[1],
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1]-5, f'{score:.2f}', 
                   color='red', fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Show predicted masks
        if len(pred_masks) > 0:
            combined_pred = torch.zeros_like(pred_masks[0, 0])
            for mask in pred_masks:
                combined_pred = torch.maximum(combined_pred, mask[0])
            ax.imshow(combined_pred.numpy(), alpha=0.3, cmap='Reds')
        
        ax.set_title(f'Predictions ({len(pred_boxes)} clouds)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{idx}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_samples} qualitative results to {save_dir}/")

# Main evaluation script
if __name__ == "__main__":
    # Import from the training script with TensorDataset
    import sys
    import os
    
    # Add parent directory to path if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from multiclass_model_v4 import create_model, Config, FilteredDataset, collate_fn
    from tensor_dataset import PackedTensorDataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    config = Config()
    model = create_model(config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please train the model first or specify correct checkpoint path")
        exit(1)
    
    model.to(device)
    
    # Create validation dataset
    print("Loading validation dataset...")
    dataset = PackedTensorDataset(tensor_path=config.tensor_path)
    
    # Create validation split
    total_samples = len(dataset)
    val_start = int(0.8 * total_samples)  # Use last 20% for validation
    val_indices = list(range(val_start, total_samples))
    
    val_dataset = FilteredDataset(dataset, indices=val_indices[:config.val_samples])
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=config.num_workers, 
        collate_fn=collate_fn
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results, predictions = evaluate_model(
        model, val_loader, device, 
        score_threshold=0.5, 
        iou_threshold=0.5
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    print(f"Average Mask IoU: {results['average_mask_iou']:.3f}")
    print(f"\nTotal Detections:")
    print(f"  True Positives: {results['total_true_positives']}")
    print(f"  False Positives: {results['total_false_positives']}")
    print(f"  False Negatives: {results['total_false_negatives']}")
    print(f"\nPer-Image Statistics:")
    print(f"  Avg Ground Truth: {results['avg_ground_truth_per_image']:.2f}")
    print(f"  Avg Predictions: {results['avg_predictions_per_image']:.2f}")
    
    # Create evaluation plots
    eval_plots_path = os.path.join(config.output_dir, 'evaluation_results.png')
    plot_evaluation_results(results, predictions, eval_plots_path)
    print(f"\nSaved evaluation plots to {eval_plots_path}")
    
    # Evaluate at multiple thresholds
    print("\nEvaluating at multiple thresholds...")
    threshold_fig, threshold_results = evaluate_at_multiple_thresholds(
        model, val_loader, device, 
        score_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    
    threshold_path = os.path.join(config.output_dir, 'threshold_analysis.png')
    threshold_fig.savefig(threshold_path, dpi=150, bbox_inches='tight')
    print(f"Saved threshold analysis to {threshold_path}")
    
    # Print best threshold
    best_threshold = max(threshold_results.keys(), 
                        key=lambda t: threshold_results[t]['f1_score'])
    print(f"\nBest threshold (by F1-score): {best_threshold}")
    print(f"  Precision: {threshold_results[best_threshold]['precision']:.3f}")
    print(f"  Recall: {threshold_results[best_threshold]['recall']:.3f}")
    print(f"  F1-Score: {threshold_results[best_threshold]['f1_score']:.3f}")
    
    # Generate qualitative results
    print("\nGenerating qualitative results...")
    qual_dir = os.path.join(config.output_dir, 'qualitative_results')
    create_qualitative_results(model, val_dataset, device, num_samples=20, save_dir=qual_dir)
    
    # Save evaluation results to JSON
    results_json_path = os.path.join(config.output_dir, 'evaluation_results.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'main_results': results,
            'threshold_analysis': {str(k): v for k, v in threshold_results.items()}
        }, f, indent=2)
    print(f"\nSaved evaluation results to {results_json_path}")
    
    print("\nEvaluation complete!")