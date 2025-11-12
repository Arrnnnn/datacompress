import cv2
import numpy as np
from scipy.fftpack import dct
from collections import Counter, namedtuple
import heapq
import pickle
import os
import time

# --- All Helper Functions from your image compressor ---
# (DCT, Huffman, RLE, Zig-Zag, Bit Packing)

ZIGZAG_INDICES = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40,
    48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63
])

def pack_bits(encoded_string):
    padding_amount = 8 - (len(encoded_string) % 8)
    if padding_amount == 8: padding_amount = 0
    padded_string = encoded_string + '0' * padding_amount
    byte_array = bytearray(int(padded_string[i:i+8], 2) for i in range(0, len(padded_string), 8))
    return bytes(byte_array), padding_amount

class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])):
    def walk(self, code, acc): self.left.walk(code, acc + "0"); self.right.walk(code, acc + "1")

class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])):
    def walk(self, code, acc): code[self.symbol] = acc or "0"

def build_huffman_tree(symbols):
    heap = [(freq, i, HuffmanLeaf(symbol)) for i, (symbol, freq) in enumerate(symbols.items())]
    heapq.heapify(heap)
    count = len(heap)
    while len(heap) > 1:
        freq1, _c1, left = heapq.heappop(heap)
        freq2, _c2, right = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, count, HuffmanNode(left, right)))
        count += 1
    if heap:
        [(_freq, _count, root)] = heap
        code = {}; root.walk(code, ""); return code
    return {}

def huffman_encode(data):
    if not data: return "", {}
    freq = Counter(data)
    code = build_huffman_tree(freq)
    return "".join(code[s] for s in data), code

def dct_2d(block): return dct(dct(block.T, norm='ortho').T, norm='ortho')

def run_length_encode(arr):
    rle_pairs = []
    zero_run = 0
    for value in arr:
        if value == 0:
            zero_run += 1
        else:
            rle_pairs.append((zero_run, value))
            zero_run = 0
    rle_pairs.append((0, 0)) # End-of-Block marker
    return rle_pairs

def compress_frame_channel(channel_data, quant_matrix):
    height, width = channel_data.shape
    all_rle_data = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel_data[i:i+8, j:j+8]
            dct_block = dct_2d(block)
            quantized_block = np.round(dct_block / quant_matrix).astype(int)
            zigzag_quant = quantized_block.flatten()[ZIGZAG_INDICES]
            rle_block_data = run_length_encode(zigzag_quant)
            all_rle_data.extend(rle_block_data)
    encoded_data, huffman_code = huffman_encode(all_rle_data)
    return encoded_data, huffman_code

# ---------------- Main Video Compression Pipeline ----------------
def main():
    input_video_path = 'input_video.mp4'
    output_compressed_path = 'compressed_video.mjpeg'
    quality = 15.0 # Adjust for video quality/size trade-off

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}"); return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Crop dimensions to be a multiple of 8
    height_cropped, width_cropped = height - (height % 8), width - (width % 8)

    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
    ]) * quality

    with open(output_compressed_path, 'wb') as f:
        # 1. Write a header with video metadata
        header = {
            'height': height_cropped, 'width': width_cropped,
            'fps': fps, 'frame_count': frame_count
        }
        pickle.dump(header, f)

        start_time = time.time()
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break

            print(f"Compressing frame {i + 1}/{frame_count}...")
            
            # Prepare frame
            frame_cropped = frame[:height_cropped, :width_cropped]
            frame_float = frame_cropped.astype(np.float64) - 128.0

            # Compress each channel and pack into bytes
            b_str, b_code = compress_frame_channel(frame_float[:,:,0], quant_matrix)
            g_str, g_code = compress_frame_channel(frame_float[:,:,1], quant_matrix)
            r_str, r_code = compress_frame_channel(frame_float[:,:,2], quant_matrix)
            
            b_data, b_pad = pack_bits(b_str)
            g_data, g_pad = pack_bits(g_str)
            r_data, r_pad = pack_bits(r_str)
            
            # 2. Write the compressed frame data to the file
            frame_package = {
                'b_code': b_code, 'g_code': g_code, 'r_code': r_code,
                'b_data': b_data, 'g_data': g_data, 'r_data': r_data,
                'b_pad': b_pad, 'g_pad': g_pad, 'r_pad': r_pad,
            }
            pickle.dump(frame_package, f)
    
    cap.release()
    end_time = time.time()
    
    original_size = os.path.getsize(input_video_path)
    compressed_size = os.path.getsize(output_compressed_path)
    
    print("\n--- Video Compression Complete ---")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Original video size: {original_size / 1024:.2f} KB")
    print(f"Compressed video size: {compressed_size / 1024:.2f} KB")
    if compressed_size > 0:
        print(f"Compression Ratio: {original_size / compressed_size:.2f}:1")
    print(f"Compressed video saved to: {output_compressed_path}")

if __name__ == '__main__':
    main()
