import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import time
from tensor_dataset import PackedTensorDataset

# -------------------- Configuration --------------------
class Config:
    # Paths
    tensor_path = "/home/mwigder/CTN/model_v4/tensorstore.pt"  # Path to tensor file or directory
    output_dir = "./cloud_segmentation_output"
    
    # Dataset
    train_samples = 600
    val_samples = 200
    
    # Model - adjusted for small objects
    num_classes = 2  # background + cloud
    backbone_layers = 5  # More layers trainable for small objects
    
    # RPN adjustments for tiny objects
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # Smaller anchors
    aspect_ratios = (( 0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    # Training
    batch_size = 1  # Small due to GPU memory
    num_workers = 4  # Reduced to avoid data loading issues
    epochs = 20  # More epochs for better convergence
    
    # Optimizer
    base_lr = 0.001  # Lower learning rate
    momentum = 0.9
    weight_decay = 1e-4
    
    # Learning rate schedule
    lr_step_size = 5
    lr_gamma = 0.5
    
    # Mixed precision
    use_amp = True
    
    # Validation
    eval_period = 1
    
    # Visualization
    visualize_predictions = True
    num_viz_samples = 5

config = Config()

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# -------------------- Enhanced Dataset Wrapper --------------------
class FilteredDataset(torch.utils.data.Dataset):
    """Wrapper to filter out None samples and add data augmentation"""
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        
        # Filter valid indices
        print("Filtering valid samples...")
        valid_indices = []
        for i in tqdm(range(len(dataset)) if indices is None else indices):
            try:
                sample = dataset[i]
                if sample is not None and len(sample) == 2:
                    image, target = sample
                    if isinstance(target, dict) and 'boxes' in target and len(target['boxes']) > 0:
                        valid_indices.append(i)
            except:
                continue
        
        self.valid_indices = valid_indices
        print(f"Found {len(self.valid_indices)} valid samples out of {len(dataset) if indices is None else len(indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.dataset[actual_idx]

# -------------------- Custom collate function --------------------
def collate_fn(batch):
    """Custom collate function that handles None values"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))

# -------------------- Model Creation --------------------
def create_model(config):
    """Create Mask R-CNN model optimized for small objects"""
    
    # Custom anchor generator for small objects
    anchor_generator = AnchorGenerator(
        sizes=config.anchor_sizes,
        aspect_ratios=config.aspect_ratios
    )
    
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers=config.backbone_layers,
        rpn_anchor_generator=anchor_generator,
        # Adjust RPN parameters for small objects
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_nms_thresh=0.5,
        box_detections_per_img=200,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25
    )
    
    # Replace the classifier heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, config.num_classes
    )
    
    return model

# -------------------- Training Functions --------------------
def train_one_epoch(model, optimizer, data_loader, device, scaler, epoch):
    model.train()
    
    # Metrics
    epoch_loss = 0.0
    loss_components = defaultdict(float)
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch} Training')
    
    for images, targets in pbar:
        # Skip empty batches
        if len(images) == 0:
            continue
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with autocast(enabled=config.use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track losses
        epoch_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    # Average losses
    num_batches = len(data_loader)
    epoch_loss /= num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return epoch_loss, dict(loss_components)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    
    # Metrics
    total_loss = 0.0
    loss_components = defaultdict(float)
    predictions = []
    
    pbar = tqdm(data_loader, desc='Validation')
    
    for images, targets in pbar:
        if len(images) == 0:
            continue
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        # Move each output dict to CPU right away
        outputs_cpu = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in o.items()} for o in outputs]
        predictions.extend(outputs_cpu)
        
        # Calculate validation loss
        model.train()
        with autocast(enabled=config.use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] += v.item()
    
    # Average losses
    num_batches = len(data_loader)
    total_loss /= num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return total_loss, dict(loss_components), predictions

# -------------------- Visualization --------------------
def visualize_predictions(images, targets, predictions, epoch, save_dir):
    """Visualize predictions vs ground truth"""
    num_samples = min(len(images), config.num_viz_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        image = images[idx].cpu()
        target = targets[idx]
        pred = predictions[idx]
        
        # Original image
        ax = axes[idx, 0]
        img_display = image[0].numpy()  # Grayscale
        ax.imshow(img_display, cmap='gray')
        ax.set_title(f'Input Image {idx}')
        ax.axis('off')
        
        # Ground truth
        ax = axes[idx, 1]
        ax.imshow(img_display, cmap='gray')
        if len(target['masks']) > 0:
            combined_mask = torch.zeros_like(target['masks'][0])
            for i, mask in enumerate(target['masks']):
                combined_mask = torch.maximum(combined_mask, mask.cpu() * (i + 1))
            ax.imshow(combined_mask.numpy(), alpha=0.5, cmap='tab20')
        ax.set_title(f'Tobac ({len(target["labels"])} objects)')
        ax.axis('off')
        
        # Predictions
        ax = axes[idx, 2]
        ax.imshow(img_display, cmap='gray')
        
        # Filter predictions by score
        score_threshold = 0.7
        keep = pred['scores'] > score_threshold
        
        if keep.sum() > 0:
            pred_masks = pred['masks'][keep].cpu()
            pred_scores = pred['scores'][keep].cpu()
            
            combined_pred = torch.zeros_like(pred_masks[0, 0])
            for i, (mask, score) in enumerate(zip(pred_masks, pred_scores)):
                # Threshold mask
                binary_mask = mask[0] > 0.5
                combined_pred = torch.maximum(combined_pred, binary_mask.float() * (i + 1))
            
            ax.imshow(combined_pred.numpy(), alpha=0.5, cmap='tab20')
            title = f'Predictions ({keep.sum()} objects)'
        else:
            title = 'Predictions (0 objects)'
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# -------------------- Main Training Loop --------------------
def main():
    # Setup
    print("torch is:", torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = PackedTensorDataset(tensor_path=config.tensor_path)
    
    # Create train/val split with filtering
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    train_indices = indices[:config.train_samples + config.val_samples + 1000]  # Extra for filtering
    val_indices = indices[config.train_samples + config.val_samples + 1000:
                         config.train_samples + config.val_samples + 2000]
    
    train_dataset = FilteredDataset(full_dataset, train_indices)
    val_dataset = FilteredDataset(full_dataset, val_indices)
    
    # Ensure we have enough samples
    train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), config.train_samples))))
    val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), config.val_samples))))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    
    # Mixed precision
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_components': [],
        'val_components': [],
        'learning_rates': []
    }
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        start_time = time.time()
        train_loss, train_components = train_one_epoch(
            model, optimizer, train_loader, device, scaler, epoch
        )
        train_time = time.time() - start_time
        
        # Validate
        if epoch % config.eval_period == 0:
            start_time = time.time()
            val_loss, val_components, predictions = evaluate(model, val_loader, device)
            val_time = time.time() - start_time
            
            # Visualize predictions
            if config.visualize_predictions:
                # Get a few samples for visualization
                viz_images, viz_targets = next(iter(val_loader))
                if len(viz_images) > 0:
                    model.eval()
                    viz_images_gpu = [img.to(device) for img in viz_images]
                    with torch.no_grad():
                        viz_predictions = model(viz_images_gpu)
                    visualize_predictions(viz_images, viz_targets, viz_predictions, 
                                        epoch, config.output_dir)
        else:
            val_loss = None
            val_components = None
            val_time = 0
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_components'].append(train_components)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        if val_loss is not None:
            history['val_loss'].append(val_loss)
            history['val_components'].append(val_components)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} (time: {train_time:.1f}s)")
        if val_loss is not None:
            print(f"Val Loss: {val_loss:.4f} (time: {val_time:.1f}s)")
            
            # Detailed loss components
            print("\nLoss Components:")
            print("  Train:", {k: f"{v:.4f}" for k, v in train_components.items()})
            print("  Val:", {k: f"{v:.4f}" for k, v in val_components.items()})
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': config.__dict__
                }, os.path.join(config.output_dir, 'best_model.pth'))
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': config.__dict__
            }, os.path.join(config.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        torch.cuda.empty_cache()

    
    # Plot training history
    plot_training_history(history, config.output_dir)
    
    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config.__dict__
    }, os.path.join(config.output_dir, 'final_model.pth'))
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {config.output_dir}")

def plot_training_history(history, save_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    if history['val_loss']:
        epochs = np.arange(1, len(history['train_loss']) + 1)
        val_epochs = epochs[::config.eval_period][:len(history['val_loss'])]
        ax.plot(val_epochs, history['val_loss'], label='Validation', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True)
    
    # Learning rate
    ax = axes[0, 1]
    ax.plot(history['learning_rates'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True)
    
    # Loss components (train)
    ax = axes[1, 0]
    if history['train_components']:
        components = history['train_components'][0].keys()
        for comp in components:
            values = [h[comp] for h in history['train_components']]
            ax.plot(values, label=comp)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True)
    
    # Loss components (validation)
    ax = axes[1, 1]
    if history['val_components']:
        components = history['val_components'][0].keys()
        for comp in components:
            values = [h[comp] for h in history['val_components']]
            ax.plot(values, label=comp)
        ax.set_xlabel('Validation Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Components')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

# -------------------- Inference Functions --------------------
@torch.no_grad()
def predict(model, image, device, score_threshold=0.5):
    """Run inference on a single image"""
    model.eval()
    
    # Prepare image
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    if image.dim() == 2:
        image = image.unsqueeze(0)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    # Get prediction
    outputs = model([image])[0]
    
    # Filter by score
    keep = outputs['scores'] > score_threshold
    
    # Return filtered predictions
    return {
        'boxes': outputs['boxes'][keep].cpu(),
        'labels': outputs['labels'][keep].cpu(),
        'scores': outputs['scores'][keep].cpu(),
        'masks': outputs['masks'][keep].cpu()
    }

def load_model_for_inference(checkpoint_path, device):
    """Load a trained model for inference"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Update config with saved values
        for key, value in saved_config.items():
            setattr(config, key, value)
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    main()