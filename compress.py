import cv2
import numpy as np
from scipy.fftpack import dct
from collections import Counter, namedtuple
import heapq
import pickle
import os

# ---------------- Constants for Zig-Zag Scan ----------------
ZIGZAG_INDICES = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])

# ---------------- Bit Packing, Huffman, and DCT (no changes) ----------------
def pack_bits(encoded_string):
    padding_amount = 8 - (len(encoded_string) % 8)
    if padding_amount == 8: padding_amount = 0
    padded_string = encoded_string + '0' * padding_amount
    byte_array = bytearray(int(padded_string[i:i+8], 2) for i in range(0, len(padded_string), 8))
    return bytes(byte_array), padding_amount

class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])):
    def walk(self, code, acc):
        code[self.symbol] = acc or "0"

def build_huffman_tree(symbols):
    # The fix is in the next line: using enumerate for the tie-breaker count
    heap = [(freq, i, HuffmanLeaf(symbol)) for i, (symbol, freq) in enumerate(symbols.items())]
    heapq.heapify(heap)
    
    count = len(heap)
    while len(heap) > 1:
        freq1, _count1, left = heapq.heappop(heap)
        freq2, _count2, right = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, count, HuffmanNode(left, right)))
        count += 1
    if heap:
        [(_freq, _count, root)] = heap
        code = {}
        root.walk(code, "")
        return code
    return {}

def huffman_encode(data):
    if not data: return "", {}
    freq = Counter(data)
    code = build_huffman_tree(freq)
    return "".join(code[s] for s in data), code

def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# ---------------- NEW Run-Length Encoding Function ----------------
def run_length_encode(arr):
    """Performs Run-Length Encoding on a 1D array."""
    rle_pairs = []
    zero_run = 0
    for value in arr:
        if value == 0:
            zero_run += 1
        else:
            rle_pairs.append((zero_run, value))
            zero_run = 0
    # Add End-of-Block marker
    rle_pairs.append((0, 0))
    return rle_pairs

# ---------------- UPDATED Compression Workflow ----------------
def compress_channel(channel_data, quant_matrix):
    height, width = channel_data.shape
    
    # Process in 8x8 blocks
    all_rle_data = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel_data[i:i+8, j:j+8]
            
            # 1. DCT and Quantize
            dct_block = dct_2d(block)
            quantized_block = np.round(dct_block / quant_matrix).astype(int)
            
            # 2. Zig-Zag Scan
            flat_quant = quantized_block.flatten()
            zigzag_quant = flat_quant[ZIGZAG_INDICES]
            
            # 3. Run-Length Encode
            rle_block_data = run_length_encode(zigzag_quant)
            all_rle_data.extend(rle_block_data)

    # 4. Huffman Encode the entire stream of RLE pairs
    encoded_data, huffman_code = huffman_encode(all_rle_data)
    
    return encoded_data, huffman_code

# ---------------- Main Compression Pipeline ----------------
def main():
    input_image_path = 'input_image.jpg'
    output_compressed_path = 'compressed_frame.dat'
    
    # Reset quality to a more reasonable value.
    # With RLE, you'll get good compression in the 10.0-30.0 range.
    quality = 5.0

    original_img = cv2.imread(input_image_path)
    if original_img is None:
        print(f"Error: Could not read image from {input_image_path}"); return

    height, width, _ = original_img.shape
    height_cropped, width_cropped = height - (height % 8), width - (width % 8)
    img_cropped = original_img[:height_cropped, :width_cropped]
    img_float = img_cropped.astype(np.float64) - 128.0

    std_quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
    ]) * quality

    print("Compressing channels with Zig-Zag and RLE...")
    b_str, b_code = compress_channel(img_float[:,:,0], std_quant_matrix)
    g_str, g_code = compress_channel(img_float[:,:,1], std_quant_matrix)
    r_str, r_code = compress_channel(img_float[:,:,2], std_quant_matrix)
    
    print("Packing bits into bytes...")
    b_data, b_pad = pack_bits(b_str)
    g_data, g_pad = pack_bits(g_str)
    r_data, r_pad = pack_bits(r_str)

    compressed_package = {
        'height': height_cropped, 'width': width_cropped,
        'b_code': b_code, 'g_code': g_code, 'r_code': r_code,
        'b_data': b_data, 'g_data': g_data, 'r_data': r_data,
        'b_pad': b_pad, 'g_pad': g_pad, 'r_pad': r_pad,
    }

    with open(output_compressed_path, 'wb') as f:
        pickle.dump(compressed_package, f)
        
    original_size = os.path.getsize(input_image_path)
    compressed_size = os.path.getsize(output_compressed_path)
    
    print("\n--- Compression Complete ---")
    print(f"Original image size: {original_size / 1024:.2f} KB")
    print(f"Compressed file size: {compressed_size / 1024:.2f} KB")
    if compressed_size > 0:
        print(f"Compression Ratio: {original_size / compressed_size:.2f}:1")
    print(f"Compressed data saved to: {output_compressed_path}")

if __name__ == '__main__':
    main()
