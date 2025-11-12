"""
Performance benchmarking examples for the compression pipeline.

This script provides comprehensive performance testing and benchmarking
utilities for evaluating compression pipeline performance.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from compression_pipeline import CompressionPipeline
from compression_pipeline.performance import PerformanceOptimizer


def benchmark_data_sizes():
    """Benchmark compression performance across different data sizes."""
    print("=== Data Size Benchmarking ===")
    
    # Define test sizes
    test_sizes = [
        ("Tiny", (16, 16)),
        ("Small", (32, 32)),
        ("Medium", (64, 64)),
        ("Large", (128, 128)),
        ("Very Large", (256, 256)),
        ("Huge", (512, 512))
    ]
    
    pipeline = CompressionPipeline(quality=0.7)
    optimizer = PerformanceOptimizer(pipeline)
    
    # Run benchmarks
    print("Running benchmarks...")
    results = optimizer.benchmark_compression_performance(test_sizes, iterations=3)
    
    # Display results
    print("\\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Size':<12} {'Shape':<12} {'Data MB':<10} {'Comp Time':<12} {'Decomp Time':<12} {'Ratio':<8} {'Throughput':<12}")
    print("-" * 80)
    
    for size_name, size_results in results['results'].items():
        stats = size_results['stats']
        print(f"{size_name:<12} {str(size_results['shape']):<12} "
              f"{size_results['data_size_mb']:<10.3f} "
              f"{stats['avg_compression_time']:<12.4f} "
              f"{stats['avg_decompression_time']:<12.4f} "
              f"{stats['avg_compression_ratio']:<8.2f} "
              f"{stats['throughput_mb_per_sec']:<12.2f}")
    
    return results


def benchmark_quality_settings():
    """Benchmark performance across different quality settings."""
    print("\\n=== Quality Settings Benchmarking ===")
    
    test_data = np.random.rand(128, 128).astype(np.float32)
    quality_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    print("Testing quality levels...")
    for quality in quality_levels:
        pipeline = CompressionPipeline(quality=quality)
        
        # Measure compression performance
        start_time = time.perf_counter()
        compressed_data, metrics = pipeline.compress_and_measure(test_data)
        total_time = time.perf_counter() - start_time
        
        results.append({
            'quality': quality,
            'compression_ratio': metrics.compression_ratio,
            'compression_time': metrics.compression_time,
            'decompression_time': metrics.decompression_time,
            'total_time': total_time,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim,
            'compressed_size': len(compressed_data.encoded_data)
        })
    
    # Display results
    print("\\nQuality Benchmark Results:")
    print("-" * 70)
    print(f"{'Quality':<8} {'Ratio':<8} {'PSNR':<8} {'SSIM':<8} {'Comp Time':<12} {'Total Time':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['quality']:<8.1f} {result['compression_ratio']:<8.2f} "
              f"{result['psnr']:<8.1f} {result['ssim']:<8.3f} "
              f"{result['compression_time']:<12.4f} {result['total_time']:<12.4f}")
    
    return results


def benchmark_block_sizes():
    """Benchmark performance across different block sizes."""
    print("\\n=== Block Size Benchmarking ===")
    
    test_data = np.random.rand(128, 128).astype(np.float32)
    block_sizes = [4, 8, 16, 32]
    
    results = []
    
    print("Testing block sizes...")
    for block_size in block_sizes:
        pipeline = CompressionPipeline(quality=0.7, block_size=block_size)
        
        # Measure performance
        start_time = time.perf_counter()
        compressed_data, metrics = pipeline.compress_and_measure(test_data)
        total_time = time.perf_counter() - start_time
        
        results.append({
            'block_size': block_size,
            'compression_ratio': metrics.compression_ratio,
            'compression_time': metrics.compression_time,
            'decompression_time': metrics.decompression_time,
            'total_time': total_time,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'throughput': test_data.nbytes / (1024 * 1024) / total_time
        })
    
    # Display results
    print("\\nBlock Size Benchmark Results:")
    print("-" * 70)
    print(f"{'Block Size':<10} {'Ratio':<8} {'PSNR':<8} {'Comp Time':<12} {'Throughput':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['block_size']}x{result['block_size']:<6} "
              f"{result['compression_ratio']:<8.2f} {result['psnr']:<8.1f} "
              f"{result['compression_time']:<12.4f} {result['throughput']:<12.2f}")
    
    return results


def benchmark_data_types():
    """Benchmark performance across different data types and patterns."""
    print("\\n=== Data Type Benchmarking ===")
    
    # Create different types of test data
    size = (64, 64)
    data_types = {
        "Random": np.random.rand(*size).astype(np.float32),
        "Constant": np.ones(size, dtype=np.float32),
        "Linear Gradient": np.outer(np.linspace(0, 1, size[0]), 
                                   np.ones(size[1])).astype(np.float32),
        "Sinusoidal": np.sin(np.outer(np.linspace(0, 4*np.pi, size[0]), 
                                     np.linspace(0, 4*np.pi, size[1]))).astype(np.float32),
        "High Frequency": np.sin(np.outer(np.linspace(0, 20*np.pi, size[0]), 
                                         np.linspace(0, 20*np.pi, size[1]))).astype(np.float32),
        "Sparse": np.zeros(size, dtype=np.float32),
        "Noisy": np.random.rand(*size).astype(np.float32) * 0.1 + 
                np.sin(np.outer(np.linspace(0, 2*np.pi, size[0]), 
                              np.linspace(0, 2*np.pi, size[1]))).astype(np.float32)
    }
    
    # Add some structure to sparse data
    data_types["Sparse"][16:32, 16:32] = 1.0
    data_types["Sparse"][40:48, 40:48] = 0.5
    
    pipeline = CompressionPipeline(quality=0.7)
    results = []
    
    print("Testing data types...")
    for data_name, data in data_types.items():
        start_time = time.perf_counter()
        compressed_data, metrics = pipeline.compress_and_measure(data)
        total_time = time.perf_counter() - start_time
        
        results.append({
            'data_type': data_name,
            'compression_ratio': metrics.compression_ratio,
            'compression_time': metrics.compression_time,
            'total_time': total_time,
            'psnr': metrics.psnr if metrics.psnr != float('inf') else 100,
            'ssim': metrics.ssim,
            'data_range': f"{np.min(data):.3f} to {np.max(data):.3f}",
            'data_std': np.std(data)
        })
    
    # Display results
    print("\\nData Type Benchmark Results:")
    print("-" * 80)
    print(f"{'Data Type':<15} {'Ratio':<8} {'PSNR':<8} {'SSIM':<8} {'Time':<8} {'Data Range':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['data_type']:<15} {result['compression_ratio']:<8.2f} "
              f"{result['psnr']:<8.1f} {result['ssim']:<8.3f} "
              f"{result['total_time']:<8.4f} {result['data_range']:<15}")
    
    return results


def memory_usage_benchmark():
    """Benchmark memory usage during compression."""
    print("\\n=== Memory Usage Benchmarking ===")
    
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        print("psutil not available for memory monitoring")
        return {}
    
    pipeline = CompressionPipeline(quality=0.7)
    test_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    
    results = []
    
    print("Testing memory usage...")
    for size in test_sizes:
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        # Create test data
        test_data = np.random.rand(*size).astype(np.float32)
        data_memory = process.memory_info().rss / (1024 * 1024)
        
        # Compress data
        compressed_data = pipeline.compress(test_data)
        compression_memory = process.memory_info().rss / (1024 * 1024)
        
        # Decompress data
        reconstructed_data = pipeline.decompress(compressed_data)
        decompression_memory = process.memory_info().rss / (1024 * 1024)
        
        # Clean up
        del test_data, compressed_data, reconstructed_data
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        results.append({
            'size': size,
            'data_size_mb': np.prod(size) * 4 / (1024 * 1024),  # float32 = 4 bytes
            'baseline_memory': baseline_memory,
            'data_memory': data_memory,
            'compression_memory': compression_memory,
            'decompression_memory': decompression_memory,
            'final_memory': final_memory,
            'peak_memory_usage': max(compression_memory, decompression_memory) - baseline_memory,
            'memory_efficiency': (max(compression_memory, decompression_memory) - baseline_memory) / (np.prod(size) * 4 / (1024 * 1024))
        })
    
    # Display results
    print("\\nMemory Usage Benchmark Results:")
    print("-" * 70)
    print(f"{'Size':<12} {'Data MB':<10} {'Peak Usage':<12} {'Efficiency':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{str(result['size']):<12} {result['data_size_mb']:<10.3f} "
              f"{result['peak_memory_usage']:<12.2f} {result['memory_efficiency']:<12.2f}")
    
    return results


def scalability_benchmark():
    """Test scalability with increasing data sizes."""
    print("\\n=== Scalability Benchmarking ===")
    
    pipeline = CompressionPipeline(quality=0.7)
    
    # Test with increasing sizes
    base_sizes = [32, 64, 128, 256, 512]
    results = []
    
    print("Testing scalability...")
    for base_size in base_sizes:
        size = (base_size, base_size)
        data_size_mb = np.prod(size) * 4 / (1024 * 1024)
        
        # Skip very large sizes if they would be too slow
        if data_size_mb > 100:  # Skip sizes larger than 100MB for demo
            print(f"  Skipping {size} (too large for demo)")
            continue
        
        test_data = np.random.rand(*size).astype(np.float32)
        
        # Measure compression time
        start_time = time.perf_counter()
        compressed_data = pipeline.compress(test_data)
        compression_time = time.perf_counter() - start_time
        
        # Measure decompression time
        start_time = time.perf_counter()
        reconstructed_data = pipeline.decompress(compressed_data)
        decompression_time = time.perf_counter() - start_time
        
        # Calculate metrics
        compression_ratio = test_data.nbytes / len(compressed_data.encoded_data)
        throughput = data_size_mb / compression_time
        
        results.append({
            'size': size,
            'data_size_mb': data_size_mb,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'total_time': compression_time + decompression_time,
            'compression_ratio': compression_ratio,
            'throughput_mb_per_sec': throughput,
            'time_per_mb': compression_time / data_size_mb
        })
        
        print(f"  {size}: {throughput:.2f} MB/s, {compression_ratio:.2f}x compression")
    
    # Analyze scalability
    if len(results) > 1:
        print("\\nScalability Analysis:")
        
        # Check if time scales linearly with data size
        sizes = [r['data_size_mb'] for r in results]
        times = [r['compression_time'] for r in results]
        
        # Simple linear regression
        n = len(sizes)
        sum_x = sum(sizes)
        sum_y = sum(times)
        sum_xy = sum(x * y for x, y in zip(sizes, times))
        sum_x2 = sum(x * x for x in sizes)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        print(f"  Time complexity: ~{slope:.4f} seconds per MB")
        print(f"  Scalability: {'Linear' if 0.001 < slope < 0.1 else 'Non-linear'}")
    
    return results


def plot_benchmark_results(size_results, quality_results, block_results, save_plots=False):
    """Plot benchmark results."""
    print("\\n=== Plotting Benchmark Results ===")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Size vs Performance
        if size_results and 'results' in size_results:
            sizes = []
            throughputs = []
            ratios = []
            
            for size_name, result in size_results['results'].items():
                sizes.append(result['data_size_mb'])
                throughputs.append(result['stats']['throughput_mb_per_sec'])
                ratios.append(result['stats']['avg_compression_ratio'])
            
            axes[0, 0].loglog(sizes, throughputs, 'bo-')
            axes[0, 0].set_xlabel('Data Size (MB)')
            axes[0, 0].set_ylabel('Throughput (MB/s)')
            axes[0, 0].set_title('Throughput vs Data Size')
            axes[0, 0].grid(True)
            
            axes[0, 1].semilogx(sizes, ratios, 'ro-')
            axes[0, 1].set_xlabel('Data Size (MB)')
            axes[0, 1].set_ylabel('Compression Ratio')
            axes[0, 1].set_title('Compression Ratio vs Data Size')
            axes[0, 1].grid(True)
        
        # Quality vs Performance
        if quality_results:
            qualities = [r['quality'] for r in quality_results]
            ratios = [r['compression_ratio'] for r in quality_results]
            times = [r['total_time'] for r in quality_results]
            psnrs = [r['psnr'] for r in quality_results]
            
            axes[0, 2].plot(qualities, ratios, 'go-')
            axes[0, 2].set_xlabel('Quality Setting')
            axes[0, 2].set_ylabel('Compression Ratio')
            axes[0, 2].set_title('Quality vs Compression Ratio')
            axes[0, 2].grid(True)
            
            axes[1, 0].plot(qualities, times, 'mo-')
            axes[1, 0].set_xlabel('Quality Setting')
            axes[1, 0].set_ylabel('Total Time (s)')
            axes[1, 0].set_title('Quality vs Processing Time')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(ratios, psnrs, 'co-')
            axes[1, 1].set_xlabel('Compression Ratio')
            axes[1, 1].set_ylabel('PSNR (dB)')
            axes[1, 1].set_title('Compression vs Quality Tradeoff')
            axes[1, 1].grid(True)
        
        # Block Size vs Performance
        if block_results:
            block_sizes = [r['block_size'] for r in block_results]
            throughputs = [r['throughput'] for r in block_results]
            
            axes[1, 2].plot(block_sizes, throughputs, 'yo-')
            axes[1, 2].set_xlabel('Block Size')
            axes[1, 2].set_ylabel('Throughput (MB/s)')
            axes[1, 2].set_title('Block Size vs Throughput')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
            print("Plots saved as 'benchmark_results.png'")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting error: {e}")


def main():
    """Run all performance benchmarks."""
    print("Compression Pipeline - Performance Benchmarks")
    print("=" * 55)
    
    # Run benchmarks
    size_results = benchmark_data_sizes()
    quality_results = benchmark_quality_settings()
    block_results = benchmark_block_sizes()
    data_type_results = benchmark_data_types()
    memory_results = memory_usage_benchmark()
    scalability_results = scalability_benchmark()
    
    # Plot results
    plot_benchmark_results(size_results, quality_results, block_results)
    
    # Performance summary
    print("\\n=== Performance Summary ===")
    
    if size_results and 'results' in size_results:
        # Find best performing size
        best_throughput = 0
        best_size = ""
        
        for size_name, result in size_results['results'].items():
            throughput = result['stats']['throughput_mb_per_sec']
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size_name
        
        print(f"Best throughput: {best_throughput:.2f} MB/s ({best_size})")
    
    if quality_results:
        # Find best quality/compression balance
        best_score = 0
        best_quality = 0
        
        for result in quality_results:
            # Simple scoring: balance compression ratio and PSNR
            score = result['compression_ratio'] * (result['psnr'] / 100)
            if score > best_score:
                best_score = score
                best_quality = result['quality']
        
        print(f"Best quality balance: {best_quality} (score: {best_score:.2f})")
    
    if data_type_results:
        # Find most compressible data type
        best_ratio = 0
        best_data_type = ""
        
        for result in data_type_results:
            if result['compression_ratio'] > best_ratio:
                best_ratio = result['compression_ratio']
                best_data_type = result['data_type']
        
        print(f"Most compressible data: {best_data_type} ({best_ratio:.2f}x)")
    
    print("\\nRecommendations:")
    print("- Use quality 0.7-0.8 for balanced performance")
    print("- Use 8x8 blocks for general purpose compression")
    print("- Structured data compresses better than random data")
    print("- Memory usage scales linearly with data size")


if __name__ == "__main__":
    main()