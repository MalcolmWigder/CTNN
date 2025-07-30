#!/usr/bin/env python3
"""
Simple script to train a new model with configurable parameters.
Users can easily modify training settings without touching the main training code.
"""

import os
import argparse
from multiclass_model_v4 import main, Config

def train_new_model(
    tensor_path=None,
    output_dir=None,
    train_samples=600,
    val_samples=200,
    epochs=20,
    batch_size=1,
    learning_rate=0.001,
    num_classes=2,
    checkpoint=None
):
    """
    Train a new model with custom parameters.
    
    Args:
        tensor_path: Path to training data tensor file
        output_dir: Where to save the trained model
        train_samples: Number of training samples to use
        val_samples: Number of validation samples
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_classes: Number of classes (including background)
        checkpoint: Path to checkpoint to resume from
    """
    
    # Update config with user parameters
    config = Config()
    
    if tensor_path:
        config.tensor_path = tensor_path
    if output_dir:
        config.output_dir = output_dir
    
    config.train_samples = train_samples
    config.val_samples = val_samples
    config.epochs = epochs
    config.batch_size = batch_size
    config.base_lr = learning_rate
    config.num_classes = num_classes
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("Training Configuration:")
    print(f"  Data: {config.tensor_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Samples: {train_samples} train, {val_samples} val")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Classes: {num_classes}")
    
    if checkpoint:
        print(f"  Resume from: {checkpoint}")
        # TODO: Add checkpoint loading logic here
    
    print("\nStarting training...")
    
    # Temporarily replace the global config and run training
    import multiclass_model_v4
    original_config = multiclass_model_v4.config
    multiclass_model_v4.config = config
    
    try:
        main()
    finally:
        # Restore original config
        multiclass_model_v4.config = original_config
    
    print(f"\nTraining complete! Model saved to: {config.output_dir}")


def main_cli():
    parser = argparse.ArgumentParser(description="Train a new segmentation model")
    
    # Data and output
    parser.add_argument("--tensor_path", 
                       default="/home/mwigder/CTN/model_v4/tensorstore.pt",
                       help="Path to training data tensor file")
    parser.add_argument("--output_dir", 
                       default="./new_model_output",
                       help="Directory to save trained model")
    
    # Training parameters
    parser.add_argument("--train_samples", type=int, default=600,
                       help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=200,
                       help="Number of validation samples")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes (including background)")
    
    # Resume training
    parser.add_argument("--checkpoint",
                       help="Path to checkpoint to resume training from")
    
    # Quick presets
    parser.add_argument("--quick", action="store_true",
                       help="Quick training (fewer samples and epochs)")
    parser.add_argument("--full", action="store_true", 
                       help="Full training (more samples and epochs)")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.quick:
        args.train_samples = 100
        args.val_samples = 50
        args.epochs = 5
        print("Using quick training preset")
    
    elif args.full:
        args.train_samples = 1000
        args.val_samples = 300
        args.epochs = 50
        print("Using full training preset")
    
    # Train the model
    train_new_model(
        tensor_path=args.tensor_path,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        checkpoint=args.checkpoint
    )


if __name__ == "__main__":
    # Example usage as a library:
    # train_new_model(
    #     tensor_path="/path/to/my_data.pt",
    #     output_dir="./my_model",
    #     train_samples=800,
    #     epochs=30,
    #     learning_rate=0.0005
    # )
    
    main_cli()