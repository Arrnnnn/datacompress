"""
Parameter tuning examples for the compression pipeline.

This script demonstrates how to optimize compression parameters for different
use cases and shows the tradeoffs between compression ratio and quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from compression_pipeline import CompressionPipeline
from compression_pipeline.performance import PerformanceOptimizer


def quality_vs_compression_tradeoff():
    """Demonstrate quality vs compression ratio tradeoff."""
    print("=== Quality vs Compression Tradeoff ===")
    
    # Create test data with some structure
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, 64), np.linspace(0, 2*np.pi, 64))
    test_data = np.sin(x) * np.cos(y) + 0.2 * np.random.rand(64, 64)
    test_data = test_data.astype(np.float32)
    
    # Test different quality levels
    quality_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = []
    
    print("Testing quality levels...")
    for quality in quality_levels:
        pipeline = CompressionPipeline(quality=quality, block_size=8)
        compressed_data, metrics = pipeline.compress_and_measure(test_data)
        
        results.append({
            'quality': quality,
            'compression_ratio': metrics.compression_ratio,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim,
            'mse': metrics.mse,
            'compressed_size': metrics.compressed_size
        })
        
        print(f"Quality {quality:.1f}: Ratio={metrics.compression_ratio:.2f}, "
              f"PSNR={metrics.psnr:.1f}dB, SSIM={metrics.ssim:.3f}")
    
    return results, test_data


def block_size_comparison():
    """Compare different block sizes."""
    print("\\n=== Block Size Comparison ===")
    
    # Create test data
    test_data = np.random.rand(64, 64).astype(np.float32)
    
    # Test different block sizes
    block_sizes = [4, 8, 16, 32]
    results = []
    
    print("Testing block sizes...")
    for block_size in block_sizes:
        pipeline = CompressionPipeline(quality=0.7, block_size=block_size)
        compressed_data, metrics = pipeline.compress_and_measure(test_data)
        
        results.append({
            'block_size': block_size,
            'compression_ratio': metrics.compression_ratio,
            'compression_time': metrics.compression_time,
            'decompression_time': metrics.decompression_time,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim
        })
        
        print(f"Block size {block_size}x{block_size}: Ratio={metrics.compression_ratio:.2f}, "
              f"Time={metrics.compression_time:.4f}s, PSNR={metrics.psnr:.1f}dB")
    
    return results


def normalization_method_comparison():
    """Compare different normalization methods."""
    print("\\n=== Normalization Method Comparison ===")
    
    # Create test data with large dynamic range
    test_data = np.random.rand(32, 32) * 1000 + 500
    test_data = test_data.astype(np.float32)
    
    # Test different normalization methods
    normalization_methods = ['minmax', 'zscore', 'none']
    results = []
    
    print("Testing normalization methods...")
    for method in normalization_methods:
        pipeline = CompressionPipeline(quality=0.7, normalization=method)
        compressed_data, metrics = pipeline.compress_and_measure(test_data)
        
        results.append({
            'method': method,
            'compression_ratio': metrics.compression_ratio,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim,
            'mse': metrics.mse
        })
        
        print(f"Normalization '{method}': Ratio={metrics.compression_ratio:.2f}, "
              f"PSNR={metrics.psnr:.1f}dB, SSIM={metrics.ssim:.3f}")
    
    return results


def automatic_parameter_optimization():
    """Demonstrate automatic parameter optimization."""
    print("\\n=== Automatic Parameter Optimization ===")
    
    # Create sample data
    sample_data = np.random.rand(48, 48).astype(np.float32)
    
    # Initialize optimizer
    pipeline = CompressionPipeline()
    optimizer = PerformanceOptimizer(pipeline)
    
    # Optimize for different scenarios
    scenarios = [
        {"name": "High Compression", "target_ratio": 4.0, "target_quality": 20.0},
        {"name": "Balanced", "target_ratio": 2.5, "target_quality": 30.0},
        {"name": "High Quality", "target_ratio": 1.5, "target_quality": 40.0}
    ]
    
    optimization_results = {}
    
    for scenario in scenarios:
        print(f"\\nOptimizing for {scenario['name']}...")
        
        results = optimizer.optimize_pipeline_parameters(
            sample_data,
            target_compression_ratio=scenario['target_ratio'],
            target_quality_threshold=scenario['target_quality']
        )
        
        best_config = results['best_configuration']
        if best_config:
            print(f"Best configuration:")
            print(f"  Quality: {best_config['quality']}")
            print(f"  Block Size: {best_config['block_size']}")
            print(f"  Normalization: {best_config['normalization']}")
            print(f"  Achieved Ratio: {best_config['metrics'].compression_ratio:.2f}")
            print(f"  Achieved PSNR: {best_config['metrics'].psnr:.1f}dB")
            print(f"  Score: {best_config['combined_score']:.3f}")
        
        optimization_results[scenario['name']] = results
    
    return optimization_results


def data_type_specific_tuning():
    """Demonstrate parameter tuning for specific data types."""
    print("\\n=== Data Type Specific Tuning ===")
    
    # Create different types of test data
    data_types = {
        "Random Noise": np.random.rand(32, 32).astype(np.float32),
        "Smooth Gradient": np.outer(np.linspace(0, 1, 32), np.linspace(0, 1, 32)).astype(np.float32),
        "High Frequency": np.sin(np.outer(np.linspace(0, 20*np.pi, 32), np.linspace(0, 20*np.pi, 32))).astype(np.float32),
        "Sparse Data": np.zeros((32, 32), dtype=np.float32)
    }
    
    # Add some sparse data points
    data_types["Sparse Data"][5:10, 5:10] = 1.0
    data_types["Sparse Data"][20:25, 20:25] = 0.5
    
    results = {}
    
    for data_name, data in data_types.items():
        print(f"\\nTuning for {data_name}:")
        
        # Test different quality levels for this data type
        best_quality = 0.5
        best_ratio = 0
        
        for quality in [0.3, 0.5, 0.7, 0.9]:
            pipeline = CompressionPipeline(quality=quality)
            compressed_data, metrics = pipeline.compress_and_measure(data)
            
            if metrics.compression_ratio > best_ratio:
                best_ratio = metrics.compression_ratio
                best_quality = quality
        
        # Test best configuration
        pipeline = CompressionPipeline(quality=best_quality)
        compressed_data, metrics = pipeline.compress_and_measure(data)
        
        results[data_name] = {
            'best_quality': best_quality,
            'compression_ratio': metrics.compression_ratio,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim
        }
        
        print(f"  Best quality: {best_quality}")
        print(f"  Compression ratio: {metrics.compression_ratio:.2f}")
        print(f"  PSNR: {metrics.psnr:.1f}dB")
    
    return results


def plot_quality_tradeoff_curves(quality_results, save_plot=False):
    """Plot quality vs compression tradeoff curves."""
    print("\\n=== Plotting Tradeoff Curves ===")
    
    try:
        qualities = [r['quality'] for r in quality_results]
        compression_ratios = [r['compression_ratio'] for r in quality_results]
        psnr_values = [r['psnr'] for r in quality_results]
        ssim_values = [r['ssim'] for r in quality_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Compression ratio vs quality
        axes[0, 0].plot(qualities, compression_ratios, 'bo-')
        axes[0, 0].set_xlabel('Quality Setting')
        axes[0, 0].set_ylabel('Compression Ratio')
        axes[0, 0].set_title('Compression Ratio vs Quality')
        axes[0, 0].grid(True)
        
        # PSNR vs quality
        axes[0, 1].plot(qualities, psnr_values, 'ro-')
        axes[0, 1].set_xlabel('Quality Setting')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR vs Quality')
        axes[0, 1].grid(True)
        
        # SSIM vs quality
        axes[1, 0].plot(qualities, ssim_values, 'go-')
        axes[1, 0].set_xlabel('Quality Setting')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('SSIM vs Quality')
        axes[1, 0].grid(True)
        
        # Compression ratio vs PSNR (Pareto frontier)
        axes[1, 1].plot(compression_ratios, psnr_values, 'mo-')
        axes[1, 1].set_xlabel('Compression Ratio')
        axes[1, 1].set_ylabel('PSNR (dB)')
        axes[1, 1].set_title('Quality vs Compression Tradeoff')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('parameter_tuning_results.png', dpi=150, bbox_inches='tight')
            print("Plot saved as 'parameter_tuning_results.png'")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting error: {e}")


def main():
    """Run all parameter tuning examples."""
    print("Compression Pipeline - Parameter Tuning Examples")
    print("=" * 55)
    
    # Quality vs compression tradeoff
    quality_results, test_data = quality_vs_compression_tradeoff()
    
    # Block size comparison
    block_results = block_size_comparison()
    
    # Normalization method comparison
    norm_results = normalization_method_comparison()
    
    # Automatic optimization
    optimization_results = automatic_parameter_optimization()
    
    # Data type specific tuning
    data_type_results = data_type_specific_tuning()
    
    # Plot results
    plot_quality_tradeoff_curves(quality_results)
    
    # Summary recommendations
    print("\\n=== Parameter Tuning Recommendations ===")
    print("\\n1. Quality Settings:")
    print("   - Low quality (0.1-0.3): Maximum compression, acceptable for previews")
    print("   - Medium quality (0.4-0.7): Balanced compression and quality")
    print("   - High quality (0.8-0.95): Minimal loss, best for archival")
    
    print("\\n2. Block Sizes:")
    print("   - 4x4: Fast processing, good for small images")
    print("   - 8x8: Standard choice, good balance")
    print("   - 16x16+: Better for large smooth regions")
    
    print("\\n3. Normalization:")
    print("   - MinMax: Best for most data types")
    print("   - Z-score: Good for data with outliers")
    print("   - None: Use when data is already normalized")
    
    print("\\n4. Data Type Specific:")
    for data_type, result in data_type_results.items():
        print(f"   - {data_type}: Quality {result['best_quality']}, "
              f"Ratio {result['compression_ratio']:.2f}")


if __name__ == "__main__":
    main()