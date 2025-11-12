"""
Basic usage examples for the compression pipeline.

This script demonstrates the fundamental operations of the compression pipeline
including compression, decompression, and metrics calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from compression_pipeline import CompressionPipeline


def basic_compression_example():
    """Demonstrate basic compression and decompression."""
    print("=== Basic Compression Example ===")
    
    # Create sample data
    data = np.random.rand(32, 32).astype(np.float32)
    print(f"Original data shape: {data.shape}")
    print(f"Original data size: {data.nbytes} bytes")
    
    # Initialize compression pipeline
    pipeline = CompressionPipeline(quality=0.8, block_size=8)
    
    # Compress data
    print("\\nCompressing data...")
    compressed_data = pipeline.compress(data)
    
    print(f"Compressed data size: {len(compressed_data.encoded_data)} bytes")
    print(f"Compression ratio: {data.nbytes / len(compressed_data.encoded_data):.2f}")
    
    # Decompress data
    print("\\nDecompressing data...")
    reconstructed_data = pipeline.decompress(compressed_data)
    
    print(f"Reconstructed data shape: {reconstructed_data.shape}")
    
    # Calculate reconstruction error
    mse = np.mean((data - reconstructed_data) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return data, reconstructed_data, compressed_data


def compression_with_metrics_example():
    """Demonstrate compression with comprehensive metrics."""
    print("\\n=== Compression with Metrics Example ===")
    
    # Create sample data with some structure
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, 64), np.linspace(0, 4*np.pi, 64))
    data = np.sin(x) * np.cos(y) + 0.1 * np.random.rand(64, 64)
    data = data.astype(np.float32)
    
    # Initialize pipeline
    pipeline = CompressionPipeline(quality=0.7)
    
    # Compress with metrics
    compressed_data, metrics = pipeline.compress_and_measure(data)
    
    # Display metrics
    print(f"\\nCompression Metrics:")
    print(f"  Compression Ratio: {metrics.compression_ratio:.2f}")
    print(f"  Original Size: {metrics.original_size} bytes")
    print(f"  Compressed Size: {metrics.compressed_size} bytes")
    print(f"  Space Savings: {(1 - 1/metrics.compression_ratio)*100:.1f}%")
    print(f"  Compression Time: {metrics.compression_time:.4f} seconds")
    print(f"  Decompression Time: {metrics.decompression_time:.4f} seconds")
    print(f"  MSE: {metrics.mse:.6f}")
    print(f"  PSNR: {metrics.psnr:.2f} dB")
    print(f"  SSIM: {metrics.ssim:.4f}")
    
    return data, compressed_data, metrics


def different_data_types_example():
    """Demonstrate compression of different data types."""
    print("\\n=== Different Data Types Example ===")
    
    pipeline = CompressionPipeline(quality=0.6)
    
    # Test different data types
    test_cases = {
        "2D Array": np.random.rand(16, 16).astype(np.float32),
        "Integer Array": np.random.randint(0, 255, size=(16, 16)).astype(np.uint8),
        "List Data": [[i + j for j in range(8)] for i in range(8)],
        "Text Data": "Hello, World! This is a test string for compression. " * 10,
        "Binary Data": bytes(range(256)) * 2,
        "Tuple Data": tuple(range(100)),
        "Scalar Integer": 42,
        "Scalar Float": 3.14159
    }
    
    results = {}
    
    for data_type, data in test_cases.items():
        try:
            print(f"\\nTesting {data_type}:")
            
            # Get original size estimate
            if hasattr(data, 'nbytes'):
                original_size = data.nbytes
            elif isinstance(data, str):
                original_size = len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                original_size = len(data)
            else:
                original_size = len(str(data))
            
            print(f"  Original size: {original_size} bytes")
            
            # Compress
            compressed_data = pipeline.compress(data)
            compressed_size = len(compressed_data.encoded_data)
            
            print(f"  Compressed size: {compressed_size} bytes")
            print(f"  Compression ratio: {original_size / compressed_size:.2f}")
            
            # Decompress
            reconstructed_data = pipeline.decompress(compressed_data)
            print(f"  Reconstruction successful: {reconstructed_data is not None}")
            
            results[data_type] = {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': original_size / compressed_size
            }
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[data_type] = {'error': str(e)}
    
    return results


def visualize_compression_results(original_data, reconstructed_data, save_plot=False):
    """Visualize compression results."""
    print("\\n=== Visualization ===")
    
    if original_data.ndim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data
        im1 = axes[0].imshow(original_data, cmap='viridis')
        axes[0].set_title('Original Data')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Reconstructed data
        im2 = axes[1].imshow(reconstructed_data, cmap='viridis')
        axes[1].set_title('Reconstructed Data')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        difference = np.abs(original_data - reconstructed_data)
        im3 = axes[2].imshow(difference, cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('compression_results.png', dpi=150, bbox_inches='tight')
            print("Plot saved as 'compression_results.png'")
        
        plt.show()
    else:
        print("Visualization only available for 2D data")


def main():
    """Run all basic examples."""
    print("Compression Pipeline - Basic Usage Examples")
    print("=" * 50)
    
    # Basic compression
    original, reconstructed, compressed = basic_compression_example()
    
    # Compression with metrics
    data_with_structure, compressed_with_metrics, metrics = compression_with_metrics_example()
    
    # Different data types
    type_results = different_data_types_example()
    
    # Visualization (if matplotlib is available)
    try:
        visualize_compression_results(original, reconstructed)
    except ImportError:
        print("\\nMatplotlib not available for visualization")
    except Exception as e:
        print(f"\\nVisualization error: {e}")
    
    # Summary
    print("\\n=== Summary ===")
    print("Successfully demonstrated:")
    print("- Basic compression and decompression")
    print("- Metrics calculation")
    print("- Multiple data type support")
    print("- Visualization of results")
    
    print("\\nData type compression results:")
    for data_type, result in type_results.items():
        if 'error' not in result:
            print(f"  {data_type}: {result['compression_ratio']:.2f}x compression")
        else:
            print(f"  {data_type}: Failed - {result['error']}")


if __name__ == "__main__":
    main()