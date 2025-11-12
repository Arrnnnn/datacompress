# Improved JPEG Image Compression Algorithm

Enhanced JPEG compression with adaptive block processing, content-aware quantization, and perceptual optimization.

## 🎯 Overview

This project implements the standard JPEG algorithm from research literature and develops seven major improvements achieving:

- **+1.39 dB PSNR** improvement
- **1.54× better** compression ratio
- **60% reduction** in blocking artifacts
- **Full color** support (YCbCr)

## 🚀 Quick Start

### Install Dependencies

```bash
pip install numpy opencv-python scipy matplotlib
```

### Run Demo

```bash
python improved_jpeg_complete.py
```

## 📊 Key Results (Quality 50)

| Metric    | Paper    | Improved | Gain     |
| --------- | -------- | -------- | -------- |
| PSNR      | 20.83 dB | 22.22 dB | +1.39 dB |
| Ratio     | 29.91:1  | 45.96:1  | 1.54×    |
| Artifacts | High     | Low      | 60% less |

## 🔧 Features

1. **Adaptive Blocks** - Variable 4×4/8×8/16×16 sizes
2. **Content-Aware Quantization** - Custom matrices per block
3. **Perceptual Optimization** - HVS-based weighting
4. **Intelligent Chroma** - Adaptive subsampling
5. **Enhanced Entropy** - Adaptive Huffman coding
6. **Full Color** - Complete YCbCr processing
7. **Parallel Processing** - Multi-threaded execution

## 📁 Main Files

- `improved_jpeg_complete.py` - Main improved algorithm (full color)
- `improved_jpeg_grayscale.py` - Improved algorithm (grayscale)
- `algorithm_comparison_final.py` - Paper vs Improved comparison
- `new1.py` - Original paper implementation

## 🎓 Usage Example

```python
from improved_jpeg_complete import ImprovedJPEGCompressor
import cv2

# Load and compress
image = cv2.imread('sample.jpg')
compressor = ImprovedJPEGCompressor(quality_factor=50)
result = compressor.compress_image(image)

print(f"PSNR: {result['psnr']:.2f} dB")
print(f"Ratio: {result['compression_ratio']:.2f}:1")
```

## 📈 Performance

**Block Distribution:**

- 4×4 blocks: 96.6% (high detail)
- 8×8 blocks: 2.4% (medium)
- 16×16 blocks: 1.0% (smooth)

**Processing Time:** 4.92s (9.5× slower but acceptable for offline use)

## 🔬 Technical Details

### Adaptive Block Selection

```python
if complexity > 100: use 4×4   # High detail
elif complexity > 50: use 8×8   # Medium
else: use 16×16                 # Smooth
```

### Content-Aware Quantization

```python
if variance > 100: scale = 0.6  # Preserve
elif variance > 50: scale = 0.7  # Balance
else: scale = 1.3                # Compress
```

## 🎯 Applications

- Web image optimization (35% bandwidth savings)
- Photo archiving (efficient storage)
- Cloud storage (reduced costs)
- Medical imaging (quality preservation)

## ⚠️ Limitations

- 9.5× slower processing (GPU acceleration planned)
- Custom format (not standard JPEG compatible)
- Optimized for photos (not graphics/text)

## 🔮 Future Work

- GPU acceleration (10-50× speedup)
- Arithmetic coding (5-10% better)
- ML-based optimization
- Real-time processing

## 📚 References

A.M. Raid et al. "JPEG Image Compression Using Discrete Cosine Transform - A Survey." IJCSES Vol.5, No.2, April 2014.

## 📞 Documentation

- `PROJECT_REPORT.md` - Complete technical report
- `PROJECT_DEFENSE_GUIDE.md` - Presentation guide
- `COMPLETE_DEMO_SCRIPT.md` - Demo instructions

---

**Made for educational and research purposes**
