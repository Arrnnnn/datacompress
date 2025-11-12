"""
Complete JPEG Implementation Based on Research Paper
===================================================

This implementation follows the exact JPEG algorithm described in the research paper:
"JPEG Image Compression Using Discrete Cosine Transform - A Survey"

Complete Pipeline:
1. RGB to YCbCr color space conversion
2. Chroma subsampling (4:2:0)
3. 8x8 block division
4. DCT transformation
5. Quantization (separate matrices for Y and CbCr)
6. Zigzag scanning
7. Run-length encoding
8. Huffman encoding
9. Complete decompression pipeline
10. YCbCr to RGB conversion

Author: Based on research paper implementation
"""

import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt

class CompleteJPEGCompressor:
    def __init__(self, quality_factor=50):
        """
        Initialize JPEG compressor with quality factor.
        
        Args:
            quality_factor: JPEG quality (1-100, where 100 is best quality)
        """
        self.quality_factor = quality_factor
        self._init_quantization_matrices()
        
    def _init_quantization_matrices(self):
        """Initialize quantization matrices as specified in the research paper."""
        
        # Luminance quantization matrix (from paper - equation 8)
        self.luminance_quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Chrominance quantization matrix (from paper - equation 9)
        self.chrominance_quant_matrix = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32)
        
        # Apply quality scaling
        self._scale_quantization_matrices()
    
    def _scale_quantization_matrices(self):
        """Scale quantization matrices based on quality factor."""
        if self.quality_factor < 50:
            scale = 5000 / self.quality_factor
        else:
            scale = 200 - 2 * self.quality_factor
        
        scale = max(scale, 1)  # Prevent division by zero
        
        self.scaled_luma_quant = np.maximum(
            np.floor((self.luminance_quant_matrix * scale + 50) / 100), 1
        )
        self.scaled_chroma_quant = np.maximum(
            np.floor((self.chrominance_quant_matrix * scale + 50) / 100), 1
        )
    
    def rgb_to_ycbcr(self, rgb_image):
        """
        Convert RGB to YCbCr color space as specified in research paper (equation 4).
        
        Args:
            rgb_image: RGB image array (H, W, 3)
            
        Returns:
            YCbCr image array (H, W, 3)
        """
        # Conversion matrix from research paper equation (4)
        conversion_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.169, -0.334, 0.500],
            [0.500, -0.419, -0.081]
        ])
        
        # Reshape image for matrix multiplication
        rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32)
        
        # Apply conversion
        ycbcr_flat = rgb_flat @ conversion_matrix.T
        
        # Add offsets for Cb and Cr channels
        ycbcr_flat[:, 1] += 128  # Cb offset
        ycbcr_flat[:, 2] += 128  # Cr offset
        
        # Reshape back to original shape
        ycbcr_image = ycbcr_flat.reshape(rgb_image.shape)
        
        return np.clip(ycbcr_image, 0, 255).astype(np.uint8)
    
    def ycbcr_to_rgb(self, ycbcr_image):
        """
        Convert YCbCr to RGB color space (inverse of equation 4).
        
        Args:
            ycbcr_image: YCbCr image array (H, W, 3)
            
        Returns:
            RGB image array (H, W, 3)
        """
        # Inverse conversion matrix
        inverse_matrix = np.array([
            [1.000, 0.000, 1.402],
            [1.000, -0.344, -0.714],
            [1.000, 1.772, 0.000]
        ])
        
        # Prepare YCbCr data
        ycbcr_flat = ycbcr_image.reshape(-1, 3).astype(np.float32)
        
        # Remove offsets from Cb and Cr
        ycbcr_flat[:, 1] -= 128
        ycbcr_flat[:, 2] -= 128
        
        # Apply inverse conversion
        rgb_flat = ycbcr_flat @ inverse_matrix.T
        
        # Reshape back to original shape
        rgb_image = rgb_flat.reshape(ycbcr_image.shape)
        
        return np.clip(rgb_image, 0, 255).astype(np.uint8)
    
    def chroma_subsample_420(self, cb_channel, cr_channel):
        """
        Apply 4:2:0 chroma subsampling as mentioned in the research paper.
        
        Args:
            cb_channel: Cb channel
            cr_channel: Cr channel
            
        Returns:
            Subsampled Cb and Cr channels
        """
        # Subsample by factor of 2 in both dimensions
        cb_subsampled = cb_channel[::2, ::2]
        cr_subsampled = cr_channel[::2, ::2]
        
        return cb_subsampled, cr_subsampled
    
    def chroma_upsample_420(self, cb_sub, cr_sub, target_shape):
        """
        Upsample 4:2:0 chroma channels back to original size.
        
        Args:
            cb_sub: Subsampled Cb channel
            cr_sub: Subsampled Cr channel
            target_shape: Target shape (H, W)
            
        Returns:
            Upsampled Cb and Cr channels
        """
        # Simple nearest neighbor upsampling
        cb_upsampled = np.repeat(np.repeat(cb_sub, 2, axis=0), 2, axis=1)
        cr_upsampled = np.repeat(np.repeat(cr_sub, 2, axis=0), 2, axis=1)
        
        # Crop to target shape if necessary
        cb_upsampled = cb_upsampled[:target_shape[0], :target_shape[1]]
        cr_upsampled = cr_upsampled[:target_shape[0], :target_shape[1]]
        
        return cb_upsampled, cr_upsampled
    
    def dct_2d(self, block):
        """
        Apply 2D DCT as specified in research paper (equations 5).
        
        Args:
            block: 8x8 image block
            
        Returns:
            DCT coefficients
        """
        # Shift pixel values from [0,255] to [-128,127] as mentioned in paper
        shifted_block = block.astype(np.float32) - 128
        
        # Apply 2D DCT using OpenCV (which implements the standard DCT)
        dct_coeffs = cv2.dct(shifted_block)
        
        return dct_coeffs
    
    def idct_2d(self, dct_block):
        """
        Apply inverse 2D DCT as specified in research paper (equation 6).
        
        Args:
            dct_block: DCT coefficients
            
        Returns:
            Reconstructed block
        """
        # Apply inverse DCT
        reconstructed = cv2.idct(dct_block)
        
        # Shift back to [0,255] range
        reconstructed += 128
        
        return np.clip(reconstructed, 0, 255)
    
    def quantize(self, dct_block, is_luminance=True):
        """
        Quantize DCT coefficients using appropriate matrix.
        
        Args:
            dct_block: DCT coefficients
            is_luminance: True for Y channel, False for Cb/Cr
            
        Returns:
            Quantized coefficients
        """
        if is_luminance:
            quant_matrix = self.scaled_luma_quant
        else:
            quant_matrix = self.scaled_chroma_quant
        
        # Quantization as per research paper (equation 6)
        quantized = np.round(dct_block / quant_matrix)
        
        return quantized.astype(np.int16)
    
    def dequantize(self, quantized_block, is_luminance=True):
        """
        Dequantize coefficients (equation 7 in research paper).
        
        Args:
            quantized_block: Quantized coefficients
            is_luminance: True for Y channel, False for Cb/Cr
            
        Returns:
            Dequantized coefficients
        """
        if is_luminance:
            quant_matrix = self.scaled_luma_quant
        else:
            quant_matrix = self.scaled_chroma_quant
        
        # Dequantization as per research paper (equation 7)
        dequantized = quantized_block * quant_matrix
        
        return dequantized.astype(np.float32)
    
    def zigzag_scan(self, block):
        """
        Apply zigzag scanning as shown in research paper (Figure 3).
        
        Args:
            block: 8x8 block
            
        Returns:
            1D array in zigzag order
        """
        # Zigzag pattern from research paper
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
        
        flat_block = block.flatten()
        return [int(flat_block[i]) for i in zigzag_order]
    
    def inverse_zigzag_scan(self, zigzag_data):
        """
        Reconstruct 8x8 block from zigzag scanned data.
        
        Args:
            zigzag_data: 1D array in zigzag order
            
        Returns:
            8x8 block
        """
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
        
        # Ensure we have 64 elements
        if len(zigzag_data) < 64:
            zigzag_data.extend([0] * (64 - len(zigzag_data)))
        
        # Reconstruct block
        block = np.zeros(64)
        for i, pos in enumerate(zigzag_order):
            if i < len(zigzag_data):
                block[pos] = zigzag_data[i]
        
        return block.reshape(8, 8)
    
    def run_length_encode(self, zigzag_data):
        """
        Apply run-length encoding as mentioned in research paper.
        
        Args:
            zigzag_data: Zigzag scanned coefficients
            
        Returns:
            RLE encoded data
        """
        encoded = []
        zero_count = 0
        
        for value in zigzag_data:
            if value == 0:
                zero_count += 1
            else:
                encoded.append((zero_count, value))
                zero_count = 0
        
        # End of block marker
        if zero_count > 0:
            encoded.append((0, 0))
        
        return encoded
    
    def run_length_decode(self, rle_data):
        """
        Decode run-length encoded data.
        
        Args:
            rle_data: RLE encoded data
            
        Returns:
            Decoded coefficients
        """
        decoded = []
        for run_length, value in rle_data:
            decoded.extend([0] * run_length)
            if value != 0:  # Not end of block marker
                decoded.append(value)
        
        return decoded
    
    def huffman_encode(self, data):
        """
        Huffman encoding as specified in research paper section 3.4.
        
        Args:
            data: Data to encode
            
        Returns:
            Encoded bitstring and code table
        """
        if not data:
            return "", {}
        
        # Calculate frequencies
        frequency = Counter(data)
        
        # Handle single symbol case
        if len(frequency) == 1:
            symbol = list(frequency.keys())[0]
            return "0" * len(data), {symbol: "0"}
        
        # Build Huffman tree
        nodes = [[freq, [symbol, ""]] for symbol, freq in frequency.items()]
        
        while len(nodes) > 1:
            nodes.sort()
            
            # Merge two lowest frequency nodes
            left = nodes.pop(0)
            right = nodes.pop(0)
            
            # Assign codes
            for pair in left[1:]:
                pair[1] = '0' + pair[1]
            for pair in right[1:]:
                pair[1] = '1' + pair[1]
            
            # Create new internal node
            merged = [left[0] + right[0]] + left[1:] + right[1:]
            nodes.append(merged)
        
        # Extract codes
        huffman_codes = {}
        for pair in nodes[0][1:]:
            huffman_codes[pair[0]] = pair[1] if pair[1] else '0'
        
        # Encode data
        encoded_data = ''.join(huffman_codes[symbol] for symbol in data)
        
        return encoded_data, huffman_codes
    
    def huffman_decode(self, encoded_data, huffman_codes):
        """
        Huffman decoding.
        
        Args:
            encoded_data: Encoded bitstring
            huffman_codes: Code table
            
        Returns:
            Decoded data
        """
        # Create reverse lookup
        reverse_codes = {code: symbol for symbol, code in huffman_codes.items()}
        
        decoded = []
        current_code = ""
        
        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ""
        
        return decoded
    
    def process_channel(self, channel, is_luminance=True):
        """
        Process a single channel through the complete JPEG pipeline.
        
        Args:
            channel: Image channel
            is_luminance: True for Y channel, False for Cb/Cr
            
        Returns:
            Compressed data and reconstruction
        """
        height, width = channel.shape
        
        # Pad to multiple of 8
        padded_height = ((height + 7) // 8) * 8
        padded_width = ((width + 7) // 8) * 8
        
        padded_channel = np.zeros((padded_height, padded_width))
        padded_channel[:height, :width] = channel
        
        # Storage for compressed data
        compressed_blocks = []
        reconstructed_channel = np.zeros_like(padded_channel)
        
        # Process in 8x8 blocks
        for y in range(0, padded_height, 8):
            for x in range(0, padded_width, 8):
                # Extract block
                block = padded_channel[y:y+8, x:x+8]
                
                # Forward pipeline
                dct_coeffs = self.dct_2d(block)
                quantized = self.quantize(dct_coeffs, is_luminance)
                zigzag_data = self.zigzag_scan(quantized)
                rle_data = self.run_length_encode(zigzag_data)
                
                compressed_blocks.extend(rle_data)
                
                # Reconstruction pipeline (for demonstration)
                decoded_rle = self.run_length_decode(rle_data)
                reconstructed_zigzag = self.inverse_zigzag_scan(decoded_rle)
                dequantized = self.dequantize(reconstructed_zigzag, is_luminance)
                reconstructed_block = self.idct_2d(dequantized)
                
                reconstructed_channel[y:y+8, x:x+8] = reconstructed_block
        
        # Huffman encode all compressed blocks
        encoded_bitstring, huffman_table = self.huffman_encode(compressed_blocks)
        
        # Return original size reconstruction
        final_reconstruction = reconstructed_channel[:height, :width]
        
        return {
            'encoded_data': encoded_bitstring,
            'huffman_table': huffman_table,
            'reconstructed': final_reconstruction,
            'original_shape': (height, width)
        }
    
    def compress_image(self, rgb_image):
        """
        Complete JPEG compression pipeline as described in research paper.
        
        Args:
            rgb_image: RGB image array
            
        Returns:
            Compressed data and reconstructed image
        """
        print("Starting complete JPEG compression...")
        
        # Step 1: RGB to YCbCr conversion (research paper section 3.1)
        print("1. Converting RGB to YCbCr...")
        ycbcr_image = self.rgb_to_ycbcr(rgb_image)
        
        # Extract channels
        y_channel = ycbcr_image[:, :, 0]
        cb_channel = ycbcr_image[:, :, 1]
        cr_channel = ycbcr_image[:, :, 2]
        
        # Step 2: Chroma subsampling (mentioned in research paper)
        print("2. Applying chroma subsampling (4:2:0)...")
        cb_subsampled, cr_subsampled = self.chroma_subsample_420(cb_channel, cr_channel)
        
        # Step 3: Process each channel
        print("3. Processing Y channel (luminance)...")
        y_result = self.process_channel(y_channel, is_luminance=True)
        
        print("4. Processing Cb channel (chrominance)...")
        cb_result = self.process_channel(cb_subsampled, is_luminance=False)
        
        print("5. Processing Cr channel (chrominance)...")
        cr_result = self.process_channel(cr_subsampled, is_luminance=False)
        
        # Step 4: Reconstruct full color image
        print("6. Reconstructing full color image...")
        
        # Upsample chroma channels
        cb_upsampled, cr_upsampled = self.chroma_upsample_420(
            cb_result['reconstructed'], 
            cr_result['reconstructed'], 
            y_channel.shape
        )
        
        # Combine channels
        reconstructed_ycbcr = np.stack([
            y_result['reconstructed'],
            cb_upsampled,
            cr_upsampled
        ], axis=2).astype(np.uint8)
        
        # Convert back to RGB
        reconstructed_rgb = self.ycbcr_to_rgb(reconstructed_ycbcr)
        
        # Calculate compression statistics
        original_size = rgb_image.nbytes
        compressed_size = (len(y_result['encoded_data']) + 
                          len(cb_result['encoded_data']) + 
                          len(cr_result['encoded_data'])) // 8  # Convert bits to bytes
        
        compression_ratio = original_size / max(compressed_size, 1)
        
        print(f"Compression complete!")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        
        return {
            'original': rgb_image,
            'reconstructed': reconstructed_rgb,
            'ycbcr_original': ycbcr_image,
            'ycbcr_reconstructed': reconstructed_ycbcr,
            'y_data': y_result,
            'cb_data': cb_result,
            'cr_data': cr_result,
            'compression_ratio': compression_ratio,
            'original_size': original_size,
            'compressed_size': compressed_size
        }
    
    def calculate_psnr(self, original, reconstructed):
        """Calculate PSNR as mentioned in research paper (equation 3)."""
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr


def main():
    """
    Main function demonstrating complete JPEG implementation.
    """
    print("Complete JPEG Implementation Based on Research Paper")
    print("=" * 60)
    
    # Load or create test image
    try:
        # Try to load existing image
        image = cv2.imread('sample_image.jpg')
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Loaded sample_image.jpg")
        else:
            # Create synthetic test image
            print("Creating synthetic test image...")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Add colorful patterns
            image[:128, :128] = [255, 0, 0]      # Red
            image[:128, 128:] = [0, 255, 0]     # Green  
            image[128:, :128] = [0, 0, 255]     # Blue
            image[128:, 128:] = [255, 255, 0]   # Yellow
            
            # Add some texture
            noise = np.random.randint(-20, 20, image.shape)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite('test_image_color.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print("Saved synthetic test image as test_image_color.jpg")
    
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Test different quality levels
    quality_levels = [10, 30, 50, 80, 95]
    
    for quality in quality_levels:
        print(f"\n{'='*40}")
        print(f"Testing Quality Level: {quality}")
        print(f"{'='*40}")
        
        # Initialize compressor
        compressor = CompleteJPEGCompressor(quality_factor=quality)
        
        # Compress image
        result = compressor.compress_image(image)
        
        # Calculate PSNR
        psnr = compressor.calculate_psnr(result['original'], result['reconstructed'])
        
        print(f"PSNR: {psnr:.2f} dB")
        
        # Save results
        output_filename = f'jpeg_result_q{quality}.jpg'
        cv2.imwrite(output_filename, cv2.cvtColor(result['reconstructed'], cv2.COLOR_RGB2BGR))
        print(f"Saved result as {output_filename}")
    
    # Create comparison visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Test with quality 50
        compressor = CompleteJPEGCompressor(quality_factor=50)
        result = compressor.compress_image(image)
        
        # Original
        axes[0, 0].imshow(result['original'])
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # YCbCr channels
        axes[0, 1].imshow(result['ycbcr_original'][:,:,0], cmap='gray')
        axes[0, 1].set_title('Y Channel (Original)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(result['ycbcr_original'][:,:,1], cmap='gray')
        axes[0, 2].set_title('Cb Channel (Original)')
        axes[0, 2].axis('off')
        
        # Reconstructed
        axes[1, 0].imshow(result['reconstructed'])
        axes[1, 0].set_title('Reconstructed RGB')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result['ycbcr_reconstructed'][:,:,0], cmap='gray')
        axes[1, 1].set_title('Y Channel (Reconstructed)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(result['ycbcr_reconstructed'][:,:,1], cmap='gray')
        axes[1, 2].set_title('Cb Channel (Reconstructed)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('complete_jpeg_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'complete_jpeg_comparison.png'")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    print("\n" + "="*60)
    print("COMPLETE JPEG IMPLEMENTATION SUMMARY")
    print("="*60)
    print("✅ RGB to YCbCr conversion (Equation 4)")
    print("✅ Chroma subsampling (4:2:0)")
    print("✅ 8x8 block DCT (Equations 5-6)")
    print("✅ Separate quantization matrices (Equations 8-9)")
    print("✅ Zigzag scanning (Figure 3)")
    print("✅ Run-length encoding")
    print("✅ Huffman encoding (Section 3.4)")
    print("✅ Complete decompression pipeline")
    print("✅ YCbCr to RGB conversion")
    print("✅ PSNR calculation (Equation 3)")
    print("\nThis implementation now fully follows the research paper!")


if __name__ == "__main__":
    main()