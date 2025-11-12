import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter, namedtuple
import heapq

# ---------------- Huffman Utilities ----------------
class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])):
    def walk(self, code, acc):
        code[self.symbol] = acc or "0"

def build_huffman_tree(symbols):
    heap = []
    for symbol, freq in symbols.items():
        heap.append((freq, len(heap), HuffmanLeaf(symbol)))
    heapq.heapify(heap)
    count = len(heap)
    while len(heap) > 1:
        freq1, _count1, left = heapq.heappop(heap)
        freq2, _count2, right = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, count, HuffmanNode(left, right)))
        count += 1
    [(_freq, _count, root)] = heap
    code = {}
    root.walk(code, "")
    return code

def huffman_encode(data):
    freq = Counter(data)
    code = build_huffman_tree(freq)
    encoded_data = "".join(code[s] for s in data)
    return encoded_data, code

def huffman_decode(encoded_data, code):
    reverse_code = {v: k for k, v in code.items()}
    decoded = []
    current = ""
    for bit in encoded_data:
        current += bit
        if current in reverse_code:
            decoded.append(reverse_code[current])
            current = ""
    return decoded

# ---------------- DCT/Compression Workflow ----------------
def block_process(channel_data, quant_matrix, use_huffman=True):
    height, width = channel_data.shape
    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            blocks.append(channel_data[i:i+8, j:j+8])
    dct_blocks = [dct_2d(block) for block in blocks]
    quantized_blocks = [np.round(dct_block / quant_matrix).astype(int) for dct_block in dct_blocks]
    flat_quant = np.hstack([block.flatten() for block in quantized_blocks])
    
    if use_huffman:
        encoded_data, code = huffman_encode(flat_quant)
        encoded_bits = len(encoded_data)
        decoded_quant = huffman_decode(encoded_data, code)
        decoded_quant = np.array(decoded_quant).reshape(-1, 64)
    else:
        encoded_bits = flat_quant.size * 8
        decoded_quant = flat_quant.reshape(-1, 64)
    
    dequantized_blocks = [decoded_quant[i].reshape(8,8) * quant_matrix for i in range(decoded_quant.shape[0])]
    reconstructed_blocks = [idct_2d(block) for block in dequantized_blocks]
    reconstructed_channel = np.zeros((height, width))
    idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            reconstructed_channel[i:i+8, j:j+8] = reconstructed_blocks[idx]
            idx += 1
    return reconstructed_channel, encoded_bits, flat_quant.size * 8

def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def psnr_color(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# ---------------- Video Compression Pipeline ----------------
input_video_path = 'input_video.mp4'  # Set your input video path
output_video_path = 'reconstructed_huffman_video.mp4'

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

width_cropped = width - (width % 8)
height_cropped = height - (height % 8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width_cropped, height_cropped))

quality = 1.0  # Adjust quantization scale for compression quality
std_quant_matrix = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
]) * quality

frame_count = 0
total_psnr = 0
total_encoded_bits = 0
total_quantized_bits = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[:height_cropped, :width_cropped, :]
    frame_float = frame.astype(float) - 128

    reconstructed_channels = []
    for c in range(3):
        channel_data = frame_float[..., c]
        recon_channel, encoded_bits, quantized_bits = block_process(channel_data, std_quant_matrix, use_huffman=False)  # Set True for full Huffman
        reconstructed_channels.append(recon_channel)
        total_encoded_bits += encoded_bits
        total_quantized_bits += quantized_bits

    reconstructed_img = np.stack(reconstructed_channels, axis=2)
    reconstructed_img = np.clip(reconstructed_img + 128, 0, 255).astype(np.uint8)

    psnr_val = psnr_color(frame[:height_cropped, :width_cropped], reconstructed_img)
    total_psnr += psnr_val
    frame_count += 1

    out.write(reconstructed_img)

    # Comment out display for speed
    # cv2.imshow('Reconstructed Frame', reconstructed_img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

avg_psnr = total_psnr / frame_count if frame_count else 0
original_bits = height_cropped * width_cropped * 3 * 8 * frame_count

print(f"Processed {frame_count} frames")
print(f"Total original bits: {original_bits}")
print(f"Total quantized bits (pre-Huffman): {total_quantized_bits}")
print(f"Total bits after Huffman encoding (or estimate): {total_encoded_bits}")
print(f"Average Compression Ratio: {original_bits / total_encoded_bits:.2f}")
print(f"Average PSNR over video: {avg_psnr:.2f} dB")
print(f"Reconstructed video saved as {output_video_path}")
