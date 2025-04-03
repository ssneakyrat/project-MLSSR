import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_config
from models.multi_band_unet import MultiBandUNet
from models.conditional_multi_band_unet import ConditionalMultiBandUNet

def estimate_model_memory(model, input_shape, batch_size=1, device='cuda'):
    """
    Estimate the memory usage of a model during inference and training
    
    Args:
        model: Model instance
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size
        device: Device to run the model on
        
    Returns:
        Dictionary with memory usage estimates
    """
    # Create dummy input
    x = torch.randn(batch_size, *input_shape, device=device)
    
    # Get model parameters size
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    
    # Convert to MB
    param_size_mb = param_size / 1024**2
    
    # Estimate forward pass memory
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        output = model(x)
        
        forward_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        # Calculate activation size
        activation_size = 0
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_size += o.nelement() * o.element_size()
        else:
            activation_size = output.nelement() * output.element_size()
        
        activation_size_mb = activation_size / 1024**2
        
        # Estimate backward pass memory (roughly 2-3x forward pass)
        backward_memory = forward_memory * 2.5
        
        # Optimizer memory (roughly 4x parameter size for Adam)
        optimizer_memory = param_size_mb * 4
        
        # Total training memory
        total_training_memory = forward_memory + backward_memory + optimizer_memory
        
        # Mixed precision memory savings (roughly 50% for activations, not for parameters)
        mixed_precision_memory = forward_memory * 0.5 + backward_memory * 0.5 + optimizer_memory
        
        return {
            "param_count": param_count,
            "param_size_mb": param_size_mb,
            "activation_size_mb": activation_size_mb,
            "forward_memory_mb": forward_memory,
            "backward_memory_mb": backward_memory,
            "optimizer_memory_mb": optimizer_memory,
            "total_training_memory_mb": total_training_memory,
            "mixed_precision_memory_mb": mixed_precision_memory
        }
    
    except Exception as e:
        print(f"Error estimating memory: {e}")
        return {
            "param_count": param_count,
            "param_size_mb": param_size_mb,
            "error": str(e)
        }

def plot_memory_usage(batch_sizes, memory_usage, title):
    """Plot memory usage across different batch sizes"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(batch_sizes, [m["total_training_memory_mb"] for m in memory_usage], 'o-', label='FP32 Training')
    plt.plot(batch_sizes, [m["mixed_precision_memory_mb"] for m in memory_usage], 's-', label='FP16 Mixed Precision')
    
    # Add horizontal line at 12GB (RTX 3060 VRAM limit)
    plt.axhline(y=12000, color='r', linestyle='--', label='RTX 3060 VRAM (12GB)')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze memory usage of models')
    parser.add_argument('--config', type=str, default='config/conditional_model.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    original_model = MultiBandUNet(config)
    conditional_model = ConditionalMultiBandUNet(config)
    
    # Move models to device
    original_model.to(device)
    conditional_model.to(device)
    
    # Get input shape
    mel_bins = config['model']['mel_bins']
    time_frames = config['model']['time_frames']
    
    # For the original model, the input is [channels, freq_bins, time_frames]
    original_input_shape = (1, mel_bins, time_frames)
    
    # For the conditional model, the input is the same
    conditional_input_shape = (1, mel_bins, time_frames)
    
    # Batch sizes to test
    batch_sizes = [1, 2, 4, 6, 8]
    
    # Analyze original model memory usage
    original_memory_usage = []
    for batch_size in batch_sizes:
        print(f"Analyzing original model with batch size {batch_size}...")
        memory = estimate_model_memory(original_model, original_input_shape, batch_size, device)
        original_memory_usage.append(memory)
        print(f"  - FP32 Training Memory: {memory['total_training_memory_mb']:.2f} MB")
        print(f"  - FP16 Mixed Precision Memory: {memory['mixed_precision_memory_mb']:.2f} MB")
    
    # Analyze conditional model memory usage
    conditional_memory_usage = []
    for batch_size in batch_sizes:
        print(f"Analyzing conditional model with batch size {batch_size}...")
        memory = estimate_model_memory(conditional_model, conditional_input_shape, batch_size, device)
        conditional_memory_usage.append(memory)
        print(f"  - FP32 Training Memory: {memory['total_training_memory_mb']:.2f} MB")
        print(f"  - FP16 Mixed Precision Memory: {memory['mixed_precision_memory_mb']:.2f} MB")
    
    # Plot memory usage
    plot_memory_usage(batch_sizes, original_memory_usage, "Original Model Memory Usage")
    plot_memory_usage(batch_sizes, conditional_memory_usage, "Conditional Model Memory Usage")
    
    # Print summary
    print("\nModel Parameter Count Comparison:")
    print(f"Original Model: {original_memory_usage[0]['param_count']:,} parameters")
    print(f"Conditional Model: {conditional_memory_usage[0]['param_count']:,} parameters")
    
    print("\nMemory Usage Summary (Batch Size = 4):")
    print(f"Original Model FP32 Training: {original_memory_usage[2]['total_training_memory_mb']:.2f} MB")
    print(f"Original Model FP16 Mixed Precision: {original_memory_usage[2]['mixed_precision_memory_mb']:.2f} MB")
    print(f"Conditional Model FP32 Training: {conditional_memory_usage[2]['total_training_memory_mb']:.2f} MB")
    print(f"Conditional Model FP16 Mixed Precision: {conditional_memory_usage[2]['mixed_precision_memory_mb']:.2f} MB")
    
    # Calculate max batch size for RTX 3060 (12GB VRAM)
    rtx_3060_vram = 12000  # MB
    
    # Find max batch size for each model
    max_batch_original_fp32 = 0
    max_batch_original_fp16 = 0
    max_batch_conditional_fp32 = 0
    max_batch_conditional_fp16 = 0
    
    for i, batch_size in enumerate(batch_sizes):
        if original_memory_usage[i]['total_training_memory_mb'] < rtx_3060_vram:
            max_batch_original_fp32 = batch_size
        if original_memory_usage[i]['mixed_precision_memory_mb'] < rtx_3060_vram:
            max_batch_original_fp16 = batch_size
        if conditional_memory_usage[i]['total_training_memory_mb'] < rtx_3060_vram:
            max_batch_conditional_fp32 = batch_size
        if conditional_memory_usage[i]['mixed_precision_memory_mb'] < rtx_3060_vram:
            max_batch_conditional_fp16 = batch_size
    
    print("\nMaximum Batch Size for RTX 3060 (12GB VRAM):")
    print(f"Original Model FP32 Training: {max_batch_original_fp32}")
    print(f"Original Model FP16 Mixed Precision: {max_batch_original_fp16}")
    print(f"Conditional Model FP32 Training: {max_batch_conditional_fp32}")
    print(f"Conditional Model FP16 Mixed Precision: {max_batch_conditional_fp16}")

if __name__ == "__main__":
    main()