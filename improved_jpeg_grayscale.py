"""
Improved JPEG Algorithm - Grayscale Only Version
================================================

This version uses ALL the improvements (adaptive blocks, content-aware quantization,
perceptual optimization) but processes ONLY the Y channel (grayscale) for fair
comparison with the paper algorithm.

This allows fair comparison:
- Paper algorithm: Grayscale, fixed blocks, fixed quantization
- This version: Grayscale, adaptive blocks, content-aware quantization
- Full version: Full color, adaptive blocks, content-aware quantization
"""

import numpy as np
import cv2
from collections import Counter
import heapq
import time

class ImprovedJPEGGrayscale:
    """
    Improved JPEG with all enhancements but grayscale only (Y channel)
    """
    
    def __init__(self, quality_factor=50):
        self.quality_factor = quality_factor
        self._init_quantization_matrices()
        
        # Thresholds for adaptive processing
        self.variance_threshold_high = 100
        self.variance_threshold_medium = 50
    
    def _init_quantization_matrices(self):
        """Initialize quantization matrices"""
        # Standard luminance quantization matrix
        self.luminance_quant_base = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Perceptual weighting matrix
        self.perceptual_weights = np.array([
            [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
            [1.1, 1.2, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5],
            [1.2, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.5, 1.8, 2.0, 2.5, 3.5, 5.0, 6.0, 7.0],
            [2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 7.0, 8.0],
            [3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ])
        
        self._scale_quantization_matrices()
    
    def _scale_quantization_matrices(self):
        """Scale quantization matrix based on quality factor"""
        if self.quality_factor < 50:
            scale = 5000 / self.quality_factor
        else:
            scale = 200 - 2 * self.quality_factor
        
        scale = max(scale, 1)
        
        self.scaled_luma_quant = np.maximum(
            np.floor((self.luminance_quant_base * scale + 50) / 100), 1
        )
    
    def determine_optimal_block_size(self, image_region):
        """
        Determine optimal block size based on content complexity
        """
        # Calculate variance
        variance = np.var(image_region)
        
        # Calculate gradient
        grad_x = np.gradient(image_region.astype(np.float32), axis=1)
        grad_y = np.gradient(image_region.astype(np.float32), axis=0)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Combined complexity
        total_complexity = variance + gradient_magnitude
        
        # Adaptive block size selection
        if total_complexity > self.variance_threshold_high:
            return 4   # High detail
        elif total_complexity > self.variance_threshold_medium:
            return 8   # Medium
        else:
            return 16  # Smooth
    
    def calculate_edge_strength(self, block):
        """Calculate edge strength in a block"""
        sobel_x = cv2.Sobel(block.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(block.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.mean(edge_magnitude)
    
    def adaptive_quantization_matrix(self, block, base_matrix):
        """
        Generate content-aware quantization matrix
        """
        # Calculate block complexity
        variance = np.var(block)
        edge_strength = self.calculate_edge_strength(block)
        
        # Adaptive scaling factor
        if variance > self.variance_threshold_high:
            scale_factor = 0.6  # Preserve detail
        elif variance > self.variance_threshold_medium:
            scale_factor = 0.7  # Balanced
        else:
            scale_factor = 1.3  # Compress more
        
        # Edge preservation factor
        edge_threshold = 20.0
        if edge_strength > edge_threshold:
            edge_factor = 0.8  # Preserve edges
        else:
            edge_factor = 1.0
        
        # Get perceptual weights for block size
        block_size = base_matrix.shape[0]
        if block_size == 4:
            perceptual_weights = self.perceptual_weights[:4, :4]
        elif block_size == 16:
            perceptual_weights = np.tile(self.perceptual_weights, (2, 2))
        else:
            perceptual_weights = self.perceptual_weights
        
        # Generate final adaptive matrix
        final_matrix = base_matrix * scale_factor * perceptual_weights * edge_factor
        
        return np.maximum(final_matrix, 1.0)
    
    def enhanced_dct_processing(self, block, block_size=8):
        """Enhanced DCT with improved precision"""
        block_centered = block.astype(np.float64) - 128.0
        
        if block_size == 4:
            padded_block = np.zeros((8, 8))
            padded_block[:4, :4] = block_centered
            dct_coeffs = cv2.dct(padded_block)
            dct_coeffs = dct_coeffs[:4, :4]
        elif block_size == 8:
            dct_coeffs = cv2.dct(block_centered)
        elif block_size == 16:
            dct_coeffs = np.zeros((16, 16))
            for i in range(0, 16, 8):
                for j in range(0, 16, 8):
                    sub_block = block_centered[i:i+8, j:j+8]
                    dct_coeffs[i:i+8, j:j+8] = cv2.dct(sub_block)
        else:
            dct_coeffs = cv2.dct(block_centered)
        
        # Coefficient thresholding
        threshold = np.std(dct_coeffs) * 0.1
        dct_coeffs = np.where(np.abs(dct_coeffs) < threshold, 0, dct_coeffs)
        
        return dct_coeffs
    
    def enhanced_idct_processing(self, dct_block, block_size=8):
        """Enhanced inverse DCT"""
        if block_size == 4:
            padded_dct = np.zeros((8, 8))
            padded_dct[:4, :4] = dct_block
            reconstructed = cv2.idct(padded_dct)
            reconstructed = reconstructed[:4, :4]
        elif block_size == 8:
            reconstructed = cv2.idct(dct_block)
        elif block_size == 16:
            reconstructed = np.zeros((16, 16))
            for i in range(0, 16, 8):
                for j in range(0, 16, 8):
                    sub_dct = dct_block[i:i+8, j:j+8]
                    reconstructed[i:i+8, j:j+8] = cv2.idct(sub_dct)
        else:
            reconstructed = cv2.idct(dct_block)
        
        reconstructed += 128
        return np.clip(reconstructed, 0, 255)
    
    def zigzag_scan(self, block):
        """Apply zigzag scanning"""
        if block.shape == (4, 4):
            zigzag_order = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        elif block.shape == (8, 8):
            zigzag_order = [
                0, 1, 5, 6, 14, 15, 27, 28,
                2, 4, 7, 13, 16, 26, 29, 42,
                3, 8, 12, 17, 25, 30, 41, 43,
                9, 11, 18, 24, 31, 40, 44, 53,
                10, 19, 23, 32, 39, 45, 52, 54,
                20, 22, 33, 38, 46, 51, 55, 60,
                21, 34, 37, 47, 50, 56, 59, 61,
                35, 36, 48, 49, 57, 58, 62, 63
            ]
        else:
            zigzag_order = list(range(block.size))
        
        flat_block = block.flatten()
        return [int(flat_block[i]) for i in zigzag_order if i < len(flat_block)]
    
    def inverse_zigzag_scan(self, zigzag_data, block_size=8):
        """Reconstruct block from zigzag data"""
        if block_size == 4:
            zigzag_order = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
            total_size = 16
        elif block_size == 8:
            zigzag_order = [
                0, 1, 5, 6, 14, 15, 27, 28,
                2, 4, 7, 13, 16, 26, 29, 42,
                3, 8, 12, 17, 25, 30, 41, 43,
                9, 11, 18, 24, 31, 40, 44, 53,
                10, 19, 23, 32, 39, 45, 52, 54,
                20, 22, 33, 38, 46, 51, 55, 60,
                21, 34, 37, 47, 50, 56, 59, 61,
                35, 36, 48, 49, 57, 58, 62, 63
            ]
            total_size = 64
        else:
            zigzag_order = list(range(256))
            total_size = 256
        
        if len(zigzag_data) < total_size:
            zigzag_data.extend([0] * (total_size - len(zigzag_data)))
        
        block = np.zeros(total_size)
        for i, pos in enumerate(zigzag_order):
            if i < len(zigzag_data):
                block[pos] = zigzag_data[i]
        
        return block.reshape(block_size, block_size)
    
    def run_length_encode(self, zigzag_data):
        """Apply run-length encoding"""
        encoded = []
        zero_count = 0
        
        for value in zigzag_data:
            if value == 0:
                zero_count += 1
            else:
                encoded.append((zero_count, value))
                zero_count = 0
        
        if zero_count > 0:
            encoded.append((0, 0))
        
        return encoded
    
    def run_length_decode(self, rle_data):
        """Decode run-length encoded data"""
        decoded = []
        for run_length, value in rle_data:
            decoded.extend([0] * run_length)
            if value != 0:
                decoded.append(value)
        
        return decoded
    
    def build_huffman_codes(self, frequencies):
        """Build Huffman codes"""
        if len(frequencies) == 1:
            symbol = list(frequencies.keys())[0]
            return {symbol: '0'}
        
        heap = []
        for i, (symbol, freq) in enumerate(frequencies.items()):
            heapq.heappush(heap, (freq, i, symbol))
        
        next_id = len(frequencies)
        while len(heap) > 1:
            freq1, id1, left = heapq.heappop(heap)
            freq2, id2, right = heapq.heappop(heap)
            merged_freq = freq1 + freq2
            heapq.heappush(heap, (merged_freq, next_id, (left, right)))
            next_id += 1
        
        if heap:
            _, _, root = heap[0]
            codes = {}
            self._extract_codes(root, "", codes)
            return codes
        else:
            return {}
    
    def _extract_codes(self, node, code, codes):
        """Extract Huffman codes from tree"""
        if isinstance(node, tuple):
            left, right = node
            self._extract_codes(left, code + '0', codes)
            self._extract_codes(right, code + '1', codes)
        else:
            codes[node] = code if code else '0'
    
    def compress_grayscale(self, rgb_image):
        """
        Compress image using improved algorithm but grayscale only
        """
        print("Starting Improved JPEG Compression (Grayscale Only)...")
        start_time = time.time()
        
        # Convert to grayscale (Y channel only)
        print("1. Converting to grayscale (Y channel)...")
        if len(rgb_image.shape) == 3:
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = rgb_image
        
        height, width = gray_image.shape
        
        # Determine block sizes
        print("2. Analyzing content for adaptive block sizes...")
        block_analysis_size = 32
        block_assignments = {}
        
        for y in range(0, height, block_analysis_size):
            for x in range(0, width, block_analysis_size):
                region = gray_image[y:min(y+block_analysis_size, height),
                                   x:min(x+block_analysis_size, width)]
                optimal_size = self.determine_optimal_block_size(region)
                
                for by in range(y, min(y+block_analysis_size, height), optimal_size):
                    for bx in range(x, min(x+block_analysis_size, width), optimal_size):
                        block_assignments[(by, bx)] = optimal_size
        
        # Process blocks
        print("3. Processing blocks with adaptive quantization...")
        all_rle_data = []
        reconstructed_image = np.zeros((height, width))
        
        for (by, bx), block_size in block_assignments.items():
            if by + block_size <= height and bx + block_size <= width:
                block = gray_image[by:by+block_size, bx:bx+block_size]
                
                # Ensure block is correct size
                if block.shape != (block_size, block_size):
                    padded_block = np.zeros((block_size, block_size))
                    padded_block[:block.shape[0], :block.shape[1]] = block
                    block = padded_block
                
                # Get base quantization matrix
                if block_size == 4:
                    base_matrix = self.scaled_luma_quant[:4, :4]
                elif block_size == 16:
                    base_matrix = np.tile(self.scaled_luma_quant, (2, 2))
                else:
                    base_matrix = self.scaled_luma_quant
                
                # Generate adaptive quantization matrix
                adaptive_quant_matrix = self.adaptive_quantization_matrix(block, base_matrix)
                
                # Forward pipeline
                dct_coeffs = self.enhanced_dct_processing(block, block_size)
                quantized = np.round(dct_coeffs / adaptive_quant_matrix).astype(np.int16)
                zigzag_data = self.zigzag_scan(quantized)
                rle_data = self.run_length_encode(zigzag_data)
                all_rle_data.extend(rle_data)
                
                # Reconstruction pipeline
                decoded_rle = self.run_length_decode(rle_data)
                reconstructed_zigzag = self.inverse_zigzag_scan(decoded_rle, block_size)
                dequantized = reconstructed_zigzag * adaptive_quant_matrix
                reconstructed_block = self.enhanced_idct_processing(dequantized, block_size)
                
                # Place in output
                end_y = min(by + block_size, height)
                end_x = min(bx + block_size, width)
                reconstructed_image[by:end_y, bx:end_x] = \
                    reconstructed_block[:end_y-by, :end_x-bx]
        
        # Entropy coding
        print("4. Applying enhanced Huffman encoding...")
        frequencies = Counter(all_rle_data)
        huffman_codes = self.build_huffman_codes(frequencies)
        
        encoded_bitstring = ""
        for symbol in all_rle_data:
            encoded_bitstring += huffman_codes.get(symbol, "0")
        
        # Calculate statistics
        processing_time = time.time() - start_time
        original_size = gray_image.nbytes
        compressed_size = len(encoded_bitstring) // 8
        compression_ratio = original_size / max(compressed_size, 1)
        
        # Calculate PSNR
        mse = np.mean((gray_image.astype(np.float64) - reconstructed_image.astype(np.float64)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Block size distribution
        block_sizes = list(block_assignments.values())
        block_distribution = Counter(block_sizes)
        total_blocks = len(block_sizes)
        
        print(f"Compression complete in {processing_time:.2f} seconds!")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"Block distribution:")
        for size in sorted(block_distribution.keys()):
            count = block_distribution[size]
            percentage = (count / total_blocks) * 100
            print(f"  {size}×{size}: {count} blocks ({percentage:.1f}%)")
        
        return {
            'original': gray_image,
            'reconstructed': reconstructed_image.astype(np.uint8),
            'compression_ratio': compression_ratio,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'psnr': psnr,
            'processing_time': processing_time,
            'block_distribution': dict(block_distribution),
            'total_blocks': total_blocks
        }


def main():
    """
    Demonstration of improved JPEG with grayscale only
    """
    print("Improved JPEG Algorithm - Grayscale Only Version")
    print("=" * 60)
    print("Features:")
    print("✅ Adaptive Block Processing (4×4, 8×8, 16×16)")
    print("✅ Content-Aware Quantization")
    print("✅ Perceptual Optimization")
    print("✅ Enhanced Entropy Coding")
    print("✅ Grayscale Only (Y channel) - Fair comparison with paper")
    print("=" * 60)
    print()
    
    # Load image
    try:
        image = cv2.imread('sample_image.jpg')
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Loaded sample_image.jpg")
        else:
            print("Creating synthetic test image...")
            image = create_test_image()
    except:
        print("Creating synthetic test image...")
        image = create_test_image()
    
    print(f"Image shape: {image.shape}")
    print()
    
    # Test different quality levels
    qualities = [30, 50, 80]
    
    for quality in qualities:
        print(f"\n{'='*60}")
        print(f"Quality {quality} - Improved Algorithm (Grayscale)")
        print(f"{'='*60}")
        
        compressor = ImprovedJPEGGrayscale(quality_factor=quality)
        result = compressor.compress_grayscale(image)
        
        # Save result
        output_filename = f'improved_grayscale_q{quality}.jpg'
        cv2.imwrite(output_filename, result['reconstructed'])
        print(f"Saved result as {output_filename}")
    
    print(f"\n{'='*60}")
    print("✓ All grayscale versions created successfully!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - improved_grayscale_q30.jpg")
    print("  - improved_grayscale_q50.jpg")
    print("  - improved_grayscale_q80.jpg")
    print("\nThese can be compared with:")
    print("  - paper_result_q*.jpg (paper algorithm, grayscale)")
    print("  - improved_result_q*.jpg (your full algorithm, color)")


def create_test_image(size=(512, 512)):
    """Create test image"""
    image = np.zeros((*size, 3), dtype=np.uint8)
    h, w = size
    
    # Smooth gradient
    for i in range(h//2):
        for j in range(w//2):
            image[i, j] = [int(255 * i / (h//2)), int(255 * j / (w//2)), 128]
    
    # Checkerboard
    for i in range(h//2):
        for j in range(w//2, w):
            if (i//8 + j//8) % 2:
                image[i, j] = [255, 255, 255]
            else:
                image[i, j] = [0, 0, 0]
    
    # Texture
    np.random.seed(42)
    noise = np.random.randint(0, 255, (h//2, w//2, 3))
    image[h//2:, 0:w//2] = noise
    
    # Color regions
    image[h//2:, w//2:] = [100, 150, 200]
    
    return image


if __name__ == "__main__":
    main()
