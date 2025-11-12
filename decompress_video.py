import cv2
import numpy as np
from scipy.fftpack import idct
import pickle
from collections import namedtuple

# --- All Helper Functions from your image decompressor ---
# (IDCT, Huffman, RLE, Zig-Zag, Bit Unpacking)

INVERSE_ZIGZAG_INDICES = np.array([
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25,
    30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48,
    49, 57, 58, 62, 63
])

def unpack_bits(byte_data, padding_amount):
    bit_string = "".join(f"{byte:08b}" for byte in byte_data)
    return bit_string[:-padding_amount] if padding_amount > 0 else bit_string

class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])): pass
class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])): pass

def huffman_decode(encoded_data, code):
    reverse_code = {v: k for k, v in code.items()}
    decoded_symbols, current_code = [], ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_code:
            decoded_symbols.append(reverse_code[current_code])
            current_code = ""
    return decoded_symbols

def idct_2d(block): return idct(idct(block.T, norm='ortho').T, norm='ortho')

def run_length_decode(rle_pairs):
    arr = []
    for num_zeros, value in rle_pairs:
        arr.extend([0] * num_zeros); arr.append(value)
    return arr

def decompress_frame_channel(encoded_data, huffman_code, quant_matrix, height, width):
    rle_data = huffman_decode(encoded_data, huffman_code)
    reconstructed_blocks, current_block_rle = [], []
    for pair in rle_data:
        if pair == (0, 0):
            zigzag_arr = run_length_decode(current_block_rle)
            padded_arr = np.zeros(64); padded_arr[:len(zigzag_arr)] = zigzag_arr
            flat_block = padded_arr[INVERSE_ZIGZAG_INDICES]
            reconstructed_blocks.append(flat_block.reshape(8, 8))
            current_block_rle = []
        else:
            current_block_rle.append(pair)

    dequantized_blocks = [block * quant_matrix for block in reconstructed_blocks]
    idct_blocks = [idct_2d(block) for block in dequantized_blocks]
    
    reconstructed_channel = np.zeros((height, width))
    idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if idx < len(idct_blocks):
                reconstructed_channel[i:i+8, j:j+8] = idct_blocks[idx]; idx += 1
    return reconstructed_channel

# ---------------- Main Video Decompression Pipeline ----------------
def main():
    compressed_file_path = 'compressed_video.mjpeg'
    output_video_path = 'reconstructed_video.mp4'
    quality = 15.0 # MUST match the quality used in compression

    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
    ]) * quality

    with open(compressed_file_path, 'rb') as f:
        # 1. Read the header
        header = pickle.load(f)
        height, width, fps, frame_count = header['height'], header['width'], header['fps'], header['frame_count']
        
        # Prepare video writer to save the output MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print("--- Video Decompression Started ---")
        
        delay = int(1000 / fps) # Delay for playback
        for i in range(frame_count):
            print(f"Decompressing frame {i + 1}/{frame_count}...")
            
            # 2. Read the data for one frame
            frame_package = pickle.load(f)
            
            # Unpack and decompress each channel
            b_str = unpack_bits(frame_package['b_data'], frame_package['b_pad'])
            g_str = unpack_bits(frame_package['g_data'], frame_package['g_pad'])
            r_str = unpack_bits(frame_package['r_data'], frame_package['r_pad'])
            
            recon_b = decompress_frame_channel(b_str, frame_package['b_code'], quant_matrix, height, width)
            recon_g = decompress_frame_channel(g_str, frame_package['g_code'], quant_matrix, height, width)
            recon_r = decompress_frame_channel(r_str, frame_package['r_code'], quant_matrix, height, width)
            
            # Reconstruct the final frame
            reconstructed_frame = np.stack([recon_b, recon_g, recon_r], axis=2)
            reconstructed_frame = np.clip(reconstructed_frame + 128.0, 0, 255).astype(np.uint8)
            
            # Display the frame
            cv2.imshow('Reconstructed Video', reconstructed_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            # Write the frame to the output MP4 file
            out.write(reconstructed_frame)
            
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDecompression complete. Reconstructed video saved to: {output_video_path}")

if __name__ == '__main__':
    main()
