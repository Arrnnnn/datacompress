# This script implements the complete JPEG compression and decompression pipeline.
# It includes:
# 1. Huffman Encoding (Final step of compression)
# 2. Huffman Decoding
# 3. Dequantization
# 4. Inverse DCT (IDCT)
# This code now processes the entire image and saves the full reconstructed file.
#
# To run this script, you must first install the required libraries:
# pip install numpy opencv-python

import numpy as np
import cv2
from collections import Counter

# Standard quantization matrices from the JPEG standard
# For simplicity, we are using the same matrix for both luminance and chrominance
# in this example, but in a real implementation, they would be different.
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def huffman_encode(data):
    """
    Performs Huffman encoding on a list of symbols.
    It builds a Huffman tree and generates the corresponding codes.
    
    Args:
        data (list): A list of symbols to be encoded.
        
    Returns:
        tuple: A tuple containing the encoded bitstring and the Huffman codes.
    """
    if not data:
        return "", {}
    
    # Calculate frequencies of each symbol
    frequency = Counter(data)
    
    # Build a list of nodes (symbol, frequency)
    nodes = [[f, [s, ""]] for s, f in frequency.items()]
    
    # Build the Huffman tree
    while len(nodes) > 1:
        # Sort nodes by frequency
        nodes.sort()
        
        # Merge the two lowest frequency nodes
        node1 = nodes.pop(0)
        node2 = nodes.pop(0)
        
        for p in node1[1:]:
            p[1] = '0' + p[1]
        for p in node2[1:]:
            p[1] = '1' + p[1]
        
        new_node = [node1[0] + node2[0]] + node1[1:] + node2[1:]
        nodes.append(new_node)
        
    # Generate the Huffman codes
    huffman_codes = {p[0]: p[1] for p in nodes[0][1:]}
    
    # Encode the original data
    encoded_data = "".join([huffman_codes[s] for s in data])
    
    return encoded_data, huffman_codes

def huffman_decode(encoded_data, huffman_codes):
    """
    Decodes a Huffman-encoded bitstring using the provided Huffman codes.
    
    Args:
        encoded_data (str): The bitstring to decode.
        huffman_codes (dict): The dictionary of Huffman codes.
        
    Returns:
        list: The decoded list of symbols.
    """
    # Create a reverse lookup dictionary for decoding
    reverse_codes = {code: symbol for symbol, code in huffman_codes.items()}
    
    decoded_data = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
            
    return decoded_data

def get_zigzag_scan():
    """Returns the zigzag scan pattern for an 8x8 matrix."""
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

def run_length_encode(data):
    """
    Applies Run-Length Encoding to the zig-zag scanned data.
    
    Args:
        data (list): The list of coefficients from zigzag scan.
        
    Returns:
        list: The RLE-encoded list of (run-length, symbol) tuples.
    """
    encoded = []
    zero_count = 0
    
    for i in range(len(data)):
        if data[i] == 0:
            zero_count += 1
        else:
            encoded.append((zero_count, data[i]))
            zero_count = 0
            
    # Handle end of block
    if zero_count > 0:
        encoded.append((0, 0)) # End of block marker
    
    return encoded

def run_length_decode(data):
    """
    Decodes a run-length encoded list.
    
    Args:
        data (list): The RLE-encoded list of tuples.
        
    Returns:
        list: The decoded list of coefficients.
    """
    decoded = []
    for run, symbol in data:
        decoded.extend([0] * run)
        if symbol != 0: # Append only if it's not the end of block marker
            decoded.append(symbol)
    
    return decoded

def dequantize_block(quantized_block, Q_matrix):
    """
    Dequantizes a block of coefficients.
    
    Args:
        quantized_block (numpy.ndarray): The quantized 8x8 block.
        Q_matrix (numpy.ndarray): The quantization matrix.
        
    Returns:
        numpy.ndarray: The dequantized 8x8 block.
    """
    return quantized_block * Q_matrix

def main():
    """
    Main function to demonstrate the complete JPEG pipeline.
    This now processes the entire image.
    """
    try:
        # Load a sample image
        # Note: You must have a file named 'sample_image.jpg' in the same directory.
        image = cv2.imread('sample_image.jpg')
        if image is None:
            print("Error: Image not found. Please provide a valid image path.")
            return

        # ---- Step 1: Pre-processing (RGB to YCbCr & Block Division) ----
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        rows, cols, _ = ycbcr_image.shape
        block_size = 8
        
        # Initialize a blank image to store the reconstructed output
        reconstructed_y_channel = np.zeros((rows, cols), dtype=np.uint8)

        print("Processing image blocks...")
        
        # Process the image in 8x8 blocks
        for r in range(0, rows, block_size):
            for c in range(0, cols, block_size):
                # Get the 8x8 block from the Y channel
                block = ycbcr_image[r:r+block_size, c:c+block_size, 0].astype(np.float32)
                
                # Shift values from [0, 255] to [-128, 127]
                block -= 128
                
                # ---- Step 2: Forward DCT and Quantization ----
                dct_block = cv2.dct(block)
                quantized_block = np.round(dct_block / Q)

                # ---- Step 3: Huffman Encoding ----
                zigzag_scan = get_zigzag_scan()
                zigzag_data = [quantized_block.flatten()[i] for i in zigzag_scan]
                rle_encoded_data = run_length_encode(zigzag_data)
                symbols = rle_encoded_data
                encoded_bitstream, huffman_codes = huffman_encode(symbols)
                
                # ---- Step 4: Decompression Pipeline ----
                decoded_rle_tuples = huffman_decode(encoded_bitstream, huffman_codes)
                decoded_zigzag_data = run_length_decode(decoded_rle_tuples)
                decoded_zigzag_data.extend([0] * (64 - len(decoded_zigzag_data)))
                
                reconstructed_block = np.zeros((block_size, block_size))
                for i in range(len(zigzag_scan)):
                    row_idx = zigzag_scan[i] // block_size
                    col_idx = zigzag_scan[i] % block_size
                    reconstructed_block[row_idx, col_idx] = decoded_zigzag_data[i]

                dequantized_block = dequantize_block(reconstructed_block, Q)
                reconstructed_idct_block = cv2.idct(dequantized_block)
                reconstructed_idct_block += 128
                
                # Place the reconstructed block back into the final image
                reconstructed_y_channel[r:r+block_size, c:c+block_size] = np.clip(reconstructed_idct_block, 0, 255).astype(np.uint8)

        print("\nImage processing complete.")
        
        # Save the original and reconstructed images to local files
        cv2.imwrite("original_image.png", ycbcr_image[:, :, 0])
        cv2.imwrite("reconstructed_image.png", reconstructed_y_channel)
        print("Original image saved as original_image.png")
        print("Reconstructed image saved as reconstructed_image.png")

        # Display the results in windows
        cv2.imshow("Original Image", ycbcr_image[:, :, 0])
        cv2.imshow("Reconstructed Image", reconstructed_y_channel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
