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
def block_process(channel_data, quant_matrix):
    """Process an image channel: DCT, quantize, entropy encode, inverse quant/DCT."""
    height, width = channel_data.shape
    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            blocks.append(channel_data[i:i+8, j:j+8])
    # DCT and quantization
    dct_blocks = [dct_2d(block) for block in blocks]
    quantized_blocks = [np.round(dct_block / quant_matrix).astype(int) for dct_block in dct_blocks]
    # Flatten for Huffman
    flat_quant = np.hstack([block.flatten() for block in quantized_blocks])
    encoded_data, code = huffman_encode(flat_quant)
    encoded_bits = len(encoded_data)
    quantized_bits = flat_quant.size * 8
    # (Optional: decode and reconstruct)
    decoded_quant = huffman_decode(encoded_data, code)
    decoded_quant = np.array(decoded_quant).reshape(-1, 64)
    dequantized_blocks = [decoded_quant[i].reshape(8,8) * quant_matrix for i in range(decoded_quant.shape[0])]
    reconstructed_blocks = [idct_2d(block) for block in dequantized_blocks]
    # Merge reconstructed blocks
    reconstructed_channel = np.zeros((height, width))
    idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            reconstructed_channel[i:i+8, j:j+8] = reconstructed_blocks[idx]
            idx += 1
    return reconstructed_channel, encoded_bits, quantized_bits

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

# ---------------- Main JPEG Simulation ----------------

# Load color image (BGR)
img = cv2.imread('slava-jamm-9tck4_gloOQ-unsplash.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("Error: Image not found or path is incorrect.")
    exit()

img = img.astype(float) - 128
height, width, channels = img.shape
height_cropped = height - (height % 8)
width_cropped = width - (width % 8)
img_cropped = img[:height_cropped, :width_cropped, :]

# Standard quantization matrix; can scale for more/less quality
quality = 1.0  # lower = more compression/loss, higher = less compression
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

# Process each channel
reconstructed_channels = []
encoded_bits_total = 0
quantized_bits_total = 0

for c in range(channels):
    channel_data = img_cropped[..., c]
    recon_channel, encoded_bits, quantized_bits = block_process(channel_data, std_quant_matrix)
    reconstructed_channels.append(recon_channel)
    encoded_bits_total += encoded_bits
    quantized_bits_total += quantized_bits
    print(f"Channel {c}: Quantized bits: {quantized_bits}, Huffman bits: {encoded_bits}, CR: {(quantized_bits / encoded_bits):.2f}")

# Stack reconstructed channels and finalize image for display
reconstructed_img = np.stack(reconstructed_channels, axis=2)
reconstructed_img = np.clip(reconstructed_img + 128, 0, 255).astype(np.uint8)
cv2.imwrite("reconstructed_huffman_color_image.jpg", reconstructed_img)

# Show PSNR and compression ratios
original_bits = height_cropped * width_cropped * channels * 8
print(f"Original bits: {original_bits}")
print(f"Total quantized bits (pre-Huffman): {quantized_bits_total}")
print(f"Total bits after Huffman: {encoded_bits_total}")
print(f"Total Compression Ratio with Huffman: {original_bits / encoded_bits_total:.2f}")

original_cropped_uint8 = (img_cropped + 128).astype(np.uint8)
psnr_value = psnr_color(original_cropped_uint8, reconstructed_img)
print(f"PSNR between original and reconstructed image: {psnr_value:.2f} dB")

cv2.imshow('Reconstructed Huffman Compressed Image', reconstructed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
