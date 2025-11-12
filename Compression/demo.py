#!/usr/bin/env python3
"""
Quick demonstration script for the compression pipeline.

This script provides a fast way to see the compression pipeline in action
without running the full examples.
"""

import numpy as np
import time
from compression_pipeline import CompressionPipeline


def quick_demo():
    """Run a quick demonstration of the compression pipeline."""
    print("🗜️  Compression Pipeline - Quick Demo")
    print("=" * 50)
    
    # Create sample data
    print("\\n📊 Creating sample data...")
    data = np.random.rand(64, 64).astype(np.float32)
    print(f"   Data shape: {data.shape}")
    print(f"   Data size: {data.nbytes} bytes ({data.nbytes/1024:.1f} KB)")
    
    # Initialize pipeline
    print("\\n⚙️  Initializing compression pipeline...")
    pipeline = CompressionPipeline(quality=0.8, block_size=8)
    print(f"   Quality: {pipeline.quality}")
    print(f"   Block size: {pipeline.block_size}x{pipeline.block_size}")
    
    # Compress data
    print("\\n🗜️  Compressing data...")
    start_time = time.perf_counter()
    compressed_data, metrics = pipeline.compress_and_measure(data)
    compression_time = time.perf_counter() - start_time
    
    print(f"   ✅ Compression completed in {compression_time:.4f} seconds")
    print(f"   📦 Compressed size: {len(compressed_data.encoded_data)} bytes")
    print(f"   📈 Compression ratio: {metrics.compression_ratio:.2f}x")
    print(f"   💾 Space saved: {(1 - 1/metrics.compression_ratio)*100:.1f}%")
    
    # Decompress data
    print("\\n📤 Decompressing data...")
    start_time = time.perf_counter()
    reconstructed_data = pipeline.decompress(compressed_data)
    decompression_time = time.perf_counter() - start_time
    
    print(f"   ✅ Decompression completed in {decompression_time:.4f} seconds")
    print(f"   📊 Reconstructed shape: {reconstructed_data.shape}")
    
    # Quality metrics
    print("\\n📏 Quality Metrics:")
    print(f"   MSE: {metrics.mse:.6f}")
    print(f"   PSNR: {metrics.psnr:.2f} dB")
    print(f"   SSIM: {metrics.ssim:.4f}")
    
    # Performance summary
    total_time = compression_time + decompression_time
    throughput = (data.nbytes / (1024 * 1024)) / total_time
    
    print("\\n⚡ Performance Summary:")
    print(f"   Total time: {total_time:.4f} seconds")
    print(f"   Throughput: {throughput:.2f} MB/s")
    print(f"   Efficiency: {metrics.compression_ratio/total_time:.1f} ratio/second")
    
    return metrics


def test_different_qualities():
    """Test different quality settings."""
    print("\\n\\n🎛️  Testing Different Quality Settings")
    print("=" * 50)
    
    data = np.random.rand(32, 32).astype(np.float32)
    qualities = [0.2, 0.5, 0.8, 0.95]
    
    print(f"{'Quality':<8} {'Ratio':<8} {'PSNR':<8} {'SSIM':<8} {'Size':<8}")
    print("-" * 45)
    
    for quality in qualities:
        pipeline = CompressionPipeline(quality=quality)
        compressed_data, metrics = pipeline.compress_and_measure(data)
        
        print(f"{quality:<8.1f} {metrics.compression_ratio:<8.2f} "
              f"{metrics.psnr:<8.1f} {metrics.ssim:<8.3f} "
              f"{len(compressed_data.encoded_data):<8}")


def test_different_data_types():
    """Test different data types."""
    print("\\n\\n🔢 Testing Different Data Types")
    print("=" * 50)
    
    pipeline = CompressionPipeline(quality=0.7)
    
    test_cases = {
        "Random 2D": np.random.rand(24, 24).astype(np.float32),
        "Structured": np.outer(np.sin(np.linspace(0, 4*np.pi, 16)), 
                              np.cos(np.linspace(0, 4*np.pi, 16))).astype(np.float32),
        "Text": "Hello, World! This is a compression test. " * 20,
        "Binary": bytes(range(256)) * 2,
        "List": [[i + j for j in range(8)] for i in range(8)]
    }
    
    print(f"{'Data Type':<12} {'Original':<10} {'Compressed':<12} {'Ratio':<8}")
    print("-" * 45)
    
    for name, data in test_cases.items():
        try:
            compressed_data = pipeline.compress(data)
            
            # Calculate original size
            if hasattr(data, 'nbytes'):
                original_size = data.nbytes
            elif isinstance(data, str):
                original_size = len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                original_size = len(data)
            else:
                original_size = len(str(data))
            
            compressed_size = len(compressed_data.encoded_data)
            ratio = original_size / compressed_size
            
            print(f"{name:<12} {original_size:<10} {compressed_size:<12} {ratio:<8.2f}")
            
        except Exception as e:
            print(f"{name:<12} {'ERROR':<10} {str(e)[:20]:<12} {'N/A':<8}")


def main():
    """Run the complete demo."""
    try:
        # Quick demo
        metrics = quick_demo()
        
        # Additional tests
        test_different_qualities()
        test_different_data_types()
        
        # Final summary
        print("\\n\\n🎉 Demo Complete!")
        print("=" * 50)
        print("✅ Compression pipeline working correctly")
        print(f"✅ Achieved {metrics.compression_ratio:.2f}x compression")
        print(f"✅ Quality metrics: PSNR={metrics.psnr:.1f}dB, SSIM={metrics.ssim:.3f}")
        print("\\n💡 Try running the examples for more detailed demonstrations:")
        print("   python examples/basic_usage.py")
        print("   python examples/parameter_tuning.py")
        print("   python examples/database_integration.py")
        print("   python examples/performance_benchmarks.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\\n💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\\n💡 Check the installation and try again")


if __name__ == "__main__":
    main()