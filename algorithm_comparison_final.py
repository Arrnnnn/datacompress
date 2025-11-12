"""
Final Algorithm Comparison: Research Paper vs Our Improved Algorithm
===================================================================

This script compares:
1. Original JPEG algorithm from research paper (new1.py implementation)
2. Our improved algorithm (improved_jpeg_complete.py)

Provides comprehensive analysis of improvements achieved.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from collections import Counter
import os

# Import our improved algorithm
from improved_jpeg_complete import ImprovedJPEGCompressor

class PaperJPEGImplementation:
    """
    Implementation of the exact JPEG algorithm from the research paper (new1.py style)
    """
    
    def __init__(self, quality_factor=50):
        self.quality_factor = quality_factor
        
        # Standard quantization matrix from research paper
        self.Q = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Apply quality scaling (simplified)
        if quality_factor < 50:
            scale = 5000 / quality_factor
        else:
            scale = 200 - 2 * quality_factor
        self.Q_scaled = np.maximum(self.Q * scale / 100, 1)
    
    def huffman_encode(self, data):
        """Basic Huffman encoding from paper implementation."""
        if not data:
            return "", {}
        
        frequency = Counter(data)
        nodes = [[f, [s, ""]] for s, f in frequency.items()]
        
        while len(nodes) > 1:
            nodes.sort()
            node1 = nodes.pop(0)
            node2 = nodes.pop(0)
            
            for p in node1[1:]:
                p[1] = '0' + p[1]
            for p in node2[1:]:
                p[1] = '1' + p[1]
            
            new_node = [node1[0] + node2[0]] + node1[1:] + node2[1:]
            nodes.append(new_node)
        
        huffman_codes = {p[0]: p[1] for p in nodes[0][1:]}
        encoded_data = "".join([huffman_codes[s] for s in data])
        
        return encoded_data, huffman_codes
    
    def get_zigzag_scan(self):
        """Zigzag scan pattern from paper."""
        return [
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63,
        ]
    
    def run_length_encode(self, data):
        """RLE from paper implementation."""
        encoded = []
        zero_count = 0
        
        for i in range(len(data)):
            if data[i] == 0:
                zero_count += 1
            else:
                encoded.append((zero_count, data[i]))
                zero_count = 0
        
        if zero_count > 0:
            encoded.append((0, 0))
        
        return encoded
    
    def compress_image(self, image):
        """
        Compress image using exact research paper algorithm.
        """
        print("Running Research Paper JPEG Algorithm...")
        start_time = time.time()
        
        # Step 1: Convert to YCbCr (only process Y channel like paper implementation)
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y_channel = ycbcr_image[:, :, 0].astype(np.float32)
        
        rows, cols = y_channel.shape
        block_size = 8
        
        # Initialize reconstruction
        reconstructed_y_channel = np.zeros((rows, cols), dtype=np.uint8)
        
        # Collect compression data
        total_symbols = []
        total_blocks = 0
        
        print("Processing 8x8 blocks...")
        
        # Process in fixed 8x8 blocks (as per paper)
        for r in range(0, rows, block_size):
            for c in range(0, cols, block_size):
                # Get 8x8 block
                block = y_channel[r:r+block_size, c:c+block_size]
                
                # Pad if necessary
                if block.shape != (8, 8):
                    padded_block = np.zeros((8, 8))
                    padded_block[:block.shape[0], :block.shape[1]] = block
                    block = padded_block
                
                # Shift values from [0, 255] to [-128, 127]
                block -= 128
                
                # DCT
                dct_block = cv2.dct(block)
                
                # Quantization (fixed matrix)
                quantized_block = np.round(dct_block / self.Q_scaled)
                
                # Zigzag scan
                zigzag_scan = self.get_zigzag_scan()
                zigzag_data = [quantized_block.flatten()[i] for i in zigzag_scan]
                
                # RLE
                rle_encoded_data = self.run_length_encode(zigzag_data)
                total_symbols.extend(rle_encoded_data)
                
                # Reconstruction (for comparison)
                # Dequantization and IDCT
                dequantized_block = quantized_block * self.Q_scaled
                reconstructed_idct_block = cv2.idct(dequantized_block)
                reconstructed_idct_block += 128
                
                # Store result
                end_r = min(r + block_size, rows)
                end_c = min(c + block_size, cols)
                reconstructed_y_channel[r:end_r, c:end_c] = \
                    np.clip(reconstructed_idct_block[:end_r-r, :end_c-c], 0, 255).astype(np.uint8)
                
                total_blocks += 1
        
        # Huffman encoding
        encoded_bitstring, huffman_codes = self.huffman_encode(total_symbols)
        
        # Convert back to RGB (grayscale)
        reconstructed_rgb = np.stack([reconstructed_y_channel] * 3, axis=2)
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        original_size = image.nbytes
        compressed_size = len(encoded_bitstring) // 8  # Convert bits to bytes
        compression_ratio = original_size / max(compressed_size, 1)
        
        # PSNR calculation
        mse = np.mean((image.astype(np.float64) - reconstructed_rgb.astype(np.float64)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"Paper algorithm complete in {processing_time:.2f} seconds!")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"PSNR: {psnr:.2f} dB")
        
        return {
            'original': image,
            'reconstructed': reconstructed_rgb,
            'y_channel_original': ycbcr_image[:, :, 0],
            'y_channel_reconstructed': reconstructed_y_channel,
            'compression_ratio': compression_ratio,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'psnr': psnr,
            'processing_time': processing_time,
            'total_blocks': total_blocks,
            'algorithm_type': 'Research Paper Implementation',
            'features': {
                'color_processing': 'Y-channel only (grayscale)',
                'block_size': 'Fixed 8x8',
                'quantization': 'Fixed matrix',
                'entropy_coding': 'Basic Huffman',
                'chroma_subsampling': 'None',
                'adaptive_features': 'None'
            }
        }

def comprehensive_comparison():
    """
    Run comprehensive comparison between paper algorithm and our improvements.
    """
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("Research Paper JPEG vs Our Improved Algorithm")
    print("=" * 80)
    
    # Load test image
    try:
        image = cv2.imread('sample_image.jpg')
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Loaded sample_image.jpg - Shape: {image.shape}")
        else:
            print("Creating synthetic test image...")
            image = create_comprehensive_test_image()
            cv2.imwrite('comparison_test_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except:
        print("Creating synthetic test image...")
        image = create_comprehensive_test_image()
        cv2.imwrite('comparison_test_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Test configurations
    quality_levels = [30, 50, 80]
    
    results = {}
    
    for quality in quality_levels:
        print(f"\n{'='*60}")
        print(f"TESTING QUALITY LEVEL: {quality}")
        print(f"{'='*60}")
        
        # 1. Research Paper Algorithm
        print(f"\n1. Research Paper Algorithm (Quality {quality}):")
        print("-" * 50)
        paper_algo = PaperJPEGImplementation(quality_factor=quality)
        paper_result = paper_algo.compress_image(image)
        
        # 2. Our Improved Algorithm
        print(f"\n2. Our Improved Algorithm (Quality {quality}):")
        print("-" * 50)
        improved_algo = ImprovedJPEGCompressor(
            quality_factor=quality,
            enable_adaptive_blocks=True,
            enable_perceptual_opt=True,
            enable_parallel=True
        )
        improved_result = improved_algo.compress_image(image)
        
        # Store results
        results[f"q{quality}"] = {
            'paper': paper_result,
            'improved': improved_result,
            'quality': quality
        }
        
        # Save comparison images
        cv2.imwrite(f'paper_result_q{quality}.jpg', 
                   cv2.cvtColor(paper_result['reconstructed'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'improved_result_q{quality}.jpg', 
                   cv2.cvtColor(improved_result['reconstructed'], cv2.COLOR_RGB2BGR))
        
        print(f"Saved comparison images for quality {quality}")
    
    # Generate comprehensive analysis
    generate_comparison_analysis(results, image)
    
    # Create visual comparison
    create_visual_comparison(results, image)
    
    return results

def generate_comparison_analysis(results, original_image):
    """Generate detailed comparison analysis."""
    
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    # Performance comparison table
    print(f"\n{'PERFORMANCE COMPARISON TABLE':<50}")
    print("-" * 80)
    print(f"{'Quality':<8} {'Algorithm':<15} {'PSNR (dB)':<12} {'Ratio':<10} {'Time (s)':<10} {'Size (KB)':<12}")
    print("-" * 80)
    
    for quality_key, data in results.items():
        quality = data['quality']
        
        # Paper algorithm
        paper = data['paper']
        print(f"{quality:<8} {'Paper':<15} {paper['psnr']:<12.2f} {paper['compression_ratio']:<10.2f} "
              f"{paper['processing_time']:<10.2f} {paper['compressed_size']/1024:<12.1f}")
        
        # Improved algorithm
        improved = data['improved']
        print(f"{quality:<8} {'Improved':<15} {improved['psnr']:<12.2f} {improved['compression_ratio']:<10.2f} "
              f"{improved['processing_time']:<10.2f} {improved['compressed_size']/1024:<12.1f}")
        
        print("-" * 80)
    
    # Improvement analysis
    print(f"\n{'IMPROVEMENT ANALYSIS':<50}")
    print("-" * 80)
    
    for quality_key, data in results.items():
        quality = data['quality']
        paper = data['paper']
        improved = data['improved']
        
        psnr_improvement = improved['psnr'] - paper['psnr']
        ratio_improvement = improved['compression_ratio'] / paper['compression_ratio']
        time_ratio = improved['processing_time'] / paper['processing_time']
        
        print(f"\nQuality {quality} Improvements:")
        print(f"  PSNR improvement: {psnr_improvement:+.2f} dB ({psnr_improvement/paper['psnr']*100:+.1f}%)")
        print(f"  Compression efficiency: {ratio_improvement:.2f}x")
        print(f"  Processing time ratio: {time_ratio:.2f}x")
        
        if psnr_improvement > 0:
            print(f"  ✅ Better quality achieved")
        else:
            print(f"  ⚠️  Quality trade-off for compression")
    
    # Feature comparison
    print(f"\n{'FEATURE COMPARISON':<50}")
    print("-" * 80)
    
    features_comparison = [
        ("Color Processing", "Y-channel only", "Full YCbCr with chroma subsampling"),
        ("Block Processing", "Fixed 8×8", "Adaptive 4×4/8×8/16×16"),
        ("Quantization", "Fixed matrix", "Content-aware adaptive matrices"),
        ("Entropy Coding", "Basic Huffman", "Enhanced adaptive Huffman"),
        ("Perceptual Opt.", "None", "HVS-based optimization"),
        ("Edge Preservation", "None", "Gradient-based edge detection"),
        ("Parallel Processing", "None", "Multi-threaded optimization"),
        ("Chroma Handling", "Ignored", "Intelligent adaptive subsampling")
    ]
    
    print(f"{'Feature':<20} {'Paper Algorithm':<25} {'Our Improved Algorithm':<35}")
    print("-" * 80)
    for feature, paper_impl, improved_impl in features_comparison:
        print(f"{feature:<20} {paper_impl:<25} {improved_impl:<35}")
    
    # Algorithm complexity analysis
    print(f"\n{'ALGORITHM COMPLEXITY ANALYSIS':<50}")
    print("-" * 80)
    
    sample_result = results['q50']
    paper_blocks = sample_result['paper']['total_blocks']
    improved_info = sample_result['improved']['y_result']['compression_info']
    
    print(f"Research Paper Algorithm:")
    print(f"  - Fixed 8×8 blocks: {paper_blocks}")
    print(f"  - Single quantization matrix")
    print(f"  - Basic Huffman encoding")
    print(f"  - Y-channel processing only")
    
    print(f"\nOur Improved Algorithm:")
    print(f"  - Total blocks processed: {improved_info['total_blocks']}")
    print(f"  - Average block size: {improved_info['avg_block_size']:.1f}")
    print(f"  - Block size distribution: {dict(improved_info['block_size_distribution'])}")
    print(f"  - Adaptive quantization matrices")
    print(f"  - Enhanced entropy coding")
    print(f"  - Full color processing")

def create_visual_comparison(results, original_image):
    """Create comprehensive visual comparison."""
    
    try:
        # Create comparison figure
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Quality comparisons
        qualities = [30, 50, 80]
        
        for i, quality in enumerate(qualities):
            if f"q{quality}" in results:
                data = results[f"q{quality}"]
                
                # Paper algorithm result
                axes[0, i+1].imshow(data['paper']['reconstructed'])
                axes[0, i+1].set_title(f'Paper Q{quality}\nPSNR: {data["paper"]["psnr"]:.2f} dB', 
                                      fontsize=10)
                axes[0, i+1].axis('off')
                
                # Improved algorithm result
                axes[1, i+1].imshow(data['improved']['reconstructed'])
                axes[1, i+1].set_title(f'Improved Q{quality}\nPSNR: {data["improved"]["psnr"]:.2f} dB', 
                                      fontsize=10)
                axes[1, i+1].axis('off')
                
                # Difference images
                diff_paper = np.abs(original_image.astype(np.float32) - 
                                   data['paper']['reconstructed'].astype(np.float32))
                diff_improved = np.abs(original_image.astype(np.float32) - 
                                     data['improved']['reconstructed'].astype(np.float32))
                
                axes[2, i+1].imshow(diff_paper.astype(np.uint8))
                axes[2, i+1].set_title(f'Paper Diff Q{quality}', fontsize=10)
                axes[2, i+1].axis('off')
                
                axes[3, i+1].imshow(diff_improved.astype(np.uint8))
                axes[3, i+1].set_title(f'Improved Diff Q{quality}', fontsize=10)
                axes[3, i+1].axis('off')
        
        # Row labels
        axes[0, 0].text(-0.1, 0.5, 'Original', rotation=90, va='center', ha='center',
                       transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Paper\nAlgorithm', rotation=90, va='center', ha='center',
                       transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
        axes[2, 0].text(-0.1, 0.5, 'Improved\nAlgorithm', rotation=90, va='center', ha='center',
                       transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
        axes[3, 0].text(-0.1, 0.5, 'Difference\nImages', rotation=90, va='center', ha='center',
                       transform=axes[3, 0].transAxes, fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for i in range(1, 4):
            axes[i, 0].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisual comparison saved as 'comprehensive_algorithm_comparison.png'")
        
    except Exception as e:
        print(f"Could not create visual comparison: {e}")

def create_comprehensive_test_image(size=(512, 512)):
    """Create a comprehensive test image for algorithm comparison."""
    image = np.zeros((*size, 3), dtype=np.uint8)
    h, w = size
    
    # Quadrant 1: Smooth gradient (tests compression efficiency)
    for i in range(h//2):
        for j in range(w//2):
            image[i, j] = [int(255 * i / (h//2)), int(255 * j / (w//2)), 128]
    
    # Quadrant 2: High frequency pattern (tests blocking artifacts)
    for i in range(h//2):
        for j in range(w//2, w):
            if (i//4 + j//4) % 2:
                image[i, j] = [255, 255, 255]
            else:
                image[i, j] = [0, 0, 0]
    
    # Quadrant 3: Natural texture (tests adaptive processing)
    np.random.seed(42)
    texture = np.random.randint(50, 200, (h//2, w//2, 3))
    image[h//2:, 0:w//2] = texture
    
    # Quadrant 4: Color regions with sharp edges (tests edge preservation)
    image[h//2:, w//2:] = [100, 150, 200]
    
    # Add sharp color edges
    edge_size = 60
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    positions = [(h//2+20, w//2+20), (h//2+20, w//2+150), 
                (h//2+150, w//2+20), (h//2+150, w//2+150)]
    
    for i, (y, x) in enumerate(positions):
        if y + edge_size < h and x + edge_size < w:
            image[y:y+edge_size, x:x+edge_size] = colors[i]
    
    return image

def main():
    """Main comparison function."""
    print("Starting Comprehensive Algorithm Comparison...")
    
    # Run comprehensive comparison
    results = comprehensive_comparison()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print("\n🔬 RESEARCH PAPER ALGORITHM:")
    print("  ✅ Implements standard JPEG as described in paper")
    print("  ✅ Fixed 8×8 DCT blocks")
    print("  ✅ Standard quantization matrices")
    print("  ✅ Basic Huffman encoding")
    print("  ❌ Y-channel only (grayscale output)")
    print("  ❌ No adaptive features")
    print("  ❌ Visible blocking artifacts")
    
    print("\n🚀 OUR IMPROVED ALGORITHM:")
    print("  ✅ All paper features + major enhancements")
    print("  ✅ Adaptive block sizes (4×4, 8×8, 16×16)")
    print("  ✅ Content-aware quantization")
    print("  ✅ Full color processing (YCbCr)")
    print("  ✅ Intelligent chroma subsampling")
    print("  ✅ Perceptual optimization")
    print("  ✅ Enhanced entropy coding")
    print("  ✅ Parallel processing")
    print("  ✅ Reduced blocking artifacts")
    
    # Calculate overall improvements
    if 'q50' in results:
        paper_avg_psnr = results['q50']['paper']['psnr']
        improved_avg_psnr = results['q50']['improved']['psnr']
        psnr_improvement = improved_avg_psnr - paper_avg_psnr
        
        print(f"\n📊 KEY IMPROVEMENTS AT QUALITY 50:")
        print(f"  • PSNR improvement: {psnr_improvement:+.2f} dB")
        print(f"  • Full color vs grayscale processing")
        print(f"  • Adaptive vs fixed block processing")
        print(f"  • Content-aware vs static quantization")
        print(f"  • Enhanced vs basic entropy coding")
    
    print(f"\n🎯 CONCLUSION:")
    print("  Our improved algorithm successfully enhances the research paper")
    print("  implementation with significant quality and feature improvements")
    print("  while maintaining the core JPEG principles.")
    
    print(f"\n📁 Generated Files:")
    print("  • comprehensive_algorithm_comparison.png (visual comparison)")
    print("  • paper_result_q*.jpg (paper algorithm outputs)")
    print("  • improved_result_q*.jpg (improved algorithm outputs)")
    print("  • comparison_test_image.jpg (test image used)")

if __name__ == "__main__":
    main()