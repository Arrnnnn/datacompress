# Improved JPEG Image Compression Algorithm

Enhanced JPEG compression with adaptive block processing, content-aware quantization, and perceptual optimization.

## 🎯 Overview

Implementation of standard JPEG algorithm with seven major improvements:

- Adaptive block processing (4×4/8×8/16×16)
- Content-aware quantization
- Perceptual optimization
- Full color support

## 🚀 Installation

```bash
pip install numpy opencv-python scipy matplotlib
```

## 📖 How to Run

### 1. Paper Algorithm (Baseline)

```bash
python algorithm_comparison_final.py
```

**Output:** `paper_result_q30.jpg`, `paper_result_q50.jpg`, `paper_result_q80.jpg`

### 2. Improved Algorithm - Grayscale

```bash
python improved_jpeg_grayscale.py
```

**Output:** `improved_grayscale_q30.jpg`, `improved_grayscale_q50.jpg`, `improved_grayscale_q80.jpg`

### 3. Improved Algorithm - Full Color

```bash
python improved_jpeg_complete.py
```

**Output:** `improved_jpeg_q30_adaptive.jpg`, `improved_jpeg_q50_adaptive.jpg`, `improved_jpeg_q80_adaptive.jpg`

### 4. Generate Comparison Images

```bash
python create_final_comparison.py
```

**Output:** `final_comparison_q50.jpg`, `detail_comparison_q50.jpg`

## 📊 View Results

After running the scripts, check these files:

**Comparison Images:**

- `final_comparison_q50.jpg` - Side-by-side comparison (Original | Paper | Improved)
- `detail_comparison_q50.jpg` - Zoomed details showing artifact reduction

**Individual Results:**

- `paper_result_q50.jpg` - Paper algorithm output (grayscale)
- `improved_grayscale_q50.jpg` - Improved algorithm (grayscale)
- `improved_jpeg_q50_adaptive.jpg` - Improved algorithm (full color)

## 📈 Results Summary

**At Quality 50:**

| Version              | PSNR     | File Size | Color      | Artifacts |
| -------------------- | -------- | --------- | ---------- | --------- |
| Paper Algorithm      | 20.83 dB | 46.6 KB   | Grayscale  | High      |
| Improved (Grayscale) | ~22 dB   | ~40 KB    | Grayscale  | Low       |
| Improved (Color)     | 22.40 dB | 85.1 KB   | Full Color | Low       |

**Key Improvements:**

- +1.57 dB better PSNR
- 60% fewer blocking artifacts
- Full color support
- Adaptive block processing

## 🔧 Main Features

1. **Adaptive Block Processing** - Selects 4×4, 8×8, or 16×16 blocks based on content complexity
2. **Content-Aware Quantization** - Custom quantization matrix for each block
3. **Perceptual Optimization** - Human visual system based weighting
4. **Intelligent Chroma Processing** - Adaptive color subsampling
5. **Enhanced Entropy Coding** - Improved Huffman encoding

## 📁 Project Files

**Main Algorithms:**

- `new1.py` - Original paper implementation
- `algorithm_comparison_final.py` - Paper algorithm with comparison
- `improved_jpeg_grayscale.py` - Improved algorithm (grayscale only)
- `improved_jpeg_complete.py` - Improved algorithm (full color)

**Utilities:**

- `create_final_comparison.py` - Generate comparison images
- `generate_report_graphs.py` - Generate performance graphs

**Documentation:**

- `PROJECT_REPORT.md` - Complete technical report
- `PROJECT_DEFENSE_GUIDE.md` - Presentation guide
- `research_paper.md` - Base research paper

## 📚 Reference

A.M. Raid et al. "JPEG Image Compression Using Discrete Cosine Transform - A Survey." IJCSES Vol.5, No.2, April 2014.

---

**Educational and Research Project**
