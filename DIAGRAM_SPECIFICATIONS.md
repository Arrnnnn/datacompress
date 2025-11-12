# Diagram Specifications for Project Report

## Detailed Instructions for Creating All Required Figures

---

## Figure 1.1: General Architecture of JPEG Compression System

**Purpose:** High-level overview of the complete JPEG system  
**Location in Report:** Chapter 1, Page 2  
**Size:** Full width (6 inches wide × 2 inches tall)

### Diagram Layout:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   INPUT     │      │ COMPRESSION │      │ COMPRESSED  │      │DECOMPRESSION│      │   OUTPUT    │
│   IMAGE     │─────▶│   PROCESS   │─────▶│    DATA     │─────▶│   PROCESS   │─────▶│   IMAGE     │
│  (RGB/Color)│      │             │      │ (Bitstream) │      │             │      │ (RGB/Color) │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
     480×520              Multiple              ~15 KB              Reverse              480×520
     748 KB               Steps                                     Steps                748 KB
```

### Elements to Include:

1. **Input Image Box:**

   - Label: "Input Image"
   - Sub-text: "RGB Color Image"
   - Dimensions: "480×520 pixels"
   - Size: "748 KB"
   - Color: Light blue background

2. **Compression Process Box:**

   - Label: "Compression Process"
   - Sub-text: "8 Steps Pipeline"
   - List: "DCT, Quantization, Entropy Coding"
   - Color: Light green background

3. **Compressed Data Box:**

   - Label: "Compressed Data"
   - Sub-text: "Bitstream"
   - Size: "~15 KB"
   - Compression: "45:1 ratio"
   - Color: Light yellow background

4. **Decompression Process Box:**

   - Label: "Decompression Process"
   - Sub-text: "Reverse Pipeline"
   - List: "Entropy Decode, Dequantize, IDCT"
   - Color: Light orange background

5. **Output Image Box:**

   - Label: "Output Image"
   - Sub-text: "Reconstructed RGB"
   - Quality: "PSNR: 22.2 dB"
   - Color: Light blue background

6. **Arrows:**
   - Thick arrows (→) between boxes
   - Label arrows with: "Encode", "Store/Transmit", "Decode"

### Caption:

"Figure 1.1: General Architecture of JPEG Compression System showing the complete pipeline from input to output with compression and decompression stages."

---

## Figure 3.1: Standard JPEG Compression Pipeline

**Purpose:** Detailed flowchart of standard JPEG algorithm  
**Location in Report:** Chapter 3, Page 10  
**Size:** Full width (6 inches wide × 8 inches tall)

### Flowchart Layout (Vertical):

```
┌──────────────────────┐
│   RGB Image Input    │
│    (480×520×3)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Color Conversion    │
│    RGB → YCbCr       │
│ Y = 0.299R + 0.587G  │
│      + 0.114B        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Chroma Subsampling   │
│      (4:2:0)         │
│ Y: Full, Cb/Cr: ½    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Block Division      │
│   8×8 Blocks         │
│  (3,900 blocks)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Level Shift        │
│  [0,255]→[-128,127]  │
│   pixel - 128        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│        DCT           │
│ Spatial → Frequency  │
│   F(u,v) = DCT[]     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Quantization       │
│ ⚠️ LOSSY STEP ⚠️     │
│ Q = round(DCT/Q_mat) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Zigzag Scanning     │
│ 2D → 1D Sequence     │
│ Low freq → High freq │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Run-Length Encoding  │
│   Compress Zeros     │
│  [(run, value), ...] │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Huffman Encoding    │
│ Variable-Length Codes│
│  Frequent → Short    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Compressed Data     │
│    (Bitstream)       │
│     ~24 KB           │
└──────────────────────┘
```

### Styling Instructions:

1. **Box Colors:**

   - Input/Output: Light blue (#E3F2FD)
   - Transform steps (DCT, Color): Light green (#E8F5E9)
   - Quantization (lossy): Light red (#FFEBEE) with warning icon
   - Encoding steps: Light yellow (#FFF9C4)

2. **Arrows:**

   - Solid black arrows
   - Width: 3pt

3. **Text:**

   - Title: Bold, 11pt
   - Subtitle: Regular, 9pt
   - Formula: Italic, 8pt

4. **Special Marking:**
   - Add ⚠️ warning symbol to Quantization box
   - Add "ONLY LOSSY STEP" label

### Caption:

"Figure 3.1: Standard JPEG Compression Pipeline showing all nine steps from RGB input to compressed bitstream. Quantization is the only lossy step in the entire pipeline."

---

## Figure 3.2: Zigzag Scanning Pattern for 8×8 Block

**Purpose:** Illustrate the zigzag scanning order  
**Location in Report:** Chapter 3, Page 11  
**Size:** 4 inches × 4 inches (square)

### Grid Layout:

```
8×8 Grid with Numbers and Arrows:

  0 ──→ 1     5 ──→ 6    14 ──→15    27 ──→28
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
  2     4     7    13    16    26    29    42
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
  3     8    12    17    25    30    41    43
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
  9    11    18    24    31    40    44    53
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
 10    19    23    32    39    45    52    54
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
 20    22    33    38    46    51    55    60
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
 21    34    37    47    50    56    59    61
  ↓   ↗   ↘   ↗   ↘   ↗   ↘   ↗   ↘   ↗
 35    36    48    49    57    58    62    63
```

### Detailed Instructions:

1. **Grid:**

   - 8×8 cells
   - Each cell: 0.5 inch × 0.5 inch
   - Border: 1pt black lines

2. **Numbers:**

   - Position 0-63 in each cell
   - Font: Bold, 10pt
   - Color: Black

3. **Arrows:**

   - Red arrows showing scan path
   - Start at position 0 (top-left)
   - End at position 63 (bottom-right)
   - Arrow style: Curved, 2pt width

4. **Highlighting:**

   - Position 0 (DC coefficient): Yellow background
   - Low frequencies (0-15): Light green tint
   - High frequencies (48-63): Light gray tint

5. **Legend:**
   - Add small legend box:
     - Yellow: DC Coefficient
     - Green: Low Frequency (Important)
     - Gray: High Frequency (Often Zero)

### Caption:

"Figure 3.2: Zigzag Scanning Pattern for 8×8 DCT Block. The pattern orders coefficients from low frequency (top-left) to high frequency (bottom-right), grouping zeros together for efficient compression."

---

## Figure 3.3: Proposed System Architecture

**Purpose:** Complete architecture of improved algorithm  
**Location in Report:** Chapter 3, Page 15  
**Size:** Full width (6.5 inches wide × 7 inches tall)

### Architecture Diagram:

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT MODULE                            │
│  • Image Loading  • Validation  • Format Conversion         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANALYSIS MODULE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Variance    │  │  Gradient    │  │   Edge       │     │
│  │ Calculation  │  │  Analysis    │  │  Detection   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                 │
│              ┌─────────────────────────┐                    │
│              │ Complexity Assessment   │                    │
│              │ Block Size Selection    │                    │
│              └─────────────────────────┘                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 COLOR PROCESSING MODULE                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ RGB→YCbCr    │→ │  Chroma      │→ │ Adaptive     │     │
│  │ Conversion   │  │  Analysis    │  │ Subsampling  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ADAPTIVE COMPRESSION MODULE                     │
│  ┌──────────────────────────────────────────────────┐      │
│  │         PARALLEL PROCESSING (4 threads)          │      │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐│      │
│  │  │Block 1 │  │Block 2 │  │Block 3 │  │Block 4 ││      │
│  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘│      │
│  └──────┼───────────┼───────────┼───────────┼──────┘      │
│         │           │           │           │              │
│         ▼           ▼           ▼           ▼              │
│  ┌──────────────────────────────────────────────────┐     │
│  │  For Each Block:                                 │     │
│  │  1. Enhanced DCT (multi-scale)                   │     │
│  │  2. Content-Aware Quantization                   │     │
│  │  3. Perceptual Optimization                      │     │
│  │  4. Zigzag Scanning                              │     │
│  │  5. Run-Length Encoding                          │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ENTROPY CODING MODULE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Adaptive    │→ │   Huffman    │→ │  Bitstream   │     │
│  │ Probability  │  │    Tree      │  │  Generation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT MODULE                             │
│  • Compressed Data  • Metadata  • Statistics                │
│  • PSNR: 22.22 dB  • Ratio: 45.96:1  • Size: 15.9 KB       │
└─────────────────────────────────────────────────────────────┘
```

### Styling:

1. **Module Colors:**

   - Input/Output: Blue (#2196F3)
   - Analysis: Green (#4CAF50)
   - Color Processing: Purple (#9C27B0)
   - Compression: Orange (#FF9800)
   - Entropy: Red (#F44336)

2. **Boxes:**

   - Main modules: Rounded corners, thick border
   - Sub-components: Square corners, thin border

3. **Arrows:**
   - Between modules: Thick (4pt)
   - Within modules: Thin (2pt)

### Caption:

"Figure 3.3: Proposed System Architecture showing all major modules and their interactions. The system includes content analysis, adaptive processing, parallel compression, and enhanced entropy coding."

---

## Figure 3.4: Adaptive Block Size Selection Flowchart

**Purpose:** Decision tree for block size selection  
**Location in Report:** Chapter 3, Page 16  
**Size:** 5 inches wide × 6 inches tall

### Flowchart:

```
                    ┌─────────────────┐
                    │  Extract 32×32  │
                    │     Region      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Calculate:    │
                    │  • Variance     │
                    │  • Gradient     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Total Complexity│
                    │ = Var + Grad    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼────────┐    │    ┌────────▼─────────┐
    │ Complexity > 100 │    │    │ Complexity < 50  │
    │  (High Detail)   │    │    │   (Smooth)       │
    └─────────┬────────┘    │    └────────┬─────────┘
              │             │              │
    ┌─────────▼────────┐    │    ┌────────▼─────────┐
    │  Use 4×4 Blocks  │    │    │ Use 16×16 Blocks │
    │                  │    │    │                  │
    │ • Preserve Detail│    │    │ • High Compress  │
    │ • Scale = 0.6    │    │    │ • Scale = 1.3    │
    │ • 96.6% of image │    │    │ • 1.0% of image  │
    └──────────────────┘    │    └──────────────────┘
                            │
                   ┌────────▼────────┐
                   │ 50 ≤ Comp ≤ 100 │
                   │ (Medium Detail) │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Use 8×8 Blocks │
                   │                 │
                   │ • Balanced      │
                   │ • Scale = 0.7   │
                   │ • 2.4% of image │
                   └─────────────────┘
```

### Styling:

1. **Decision Diamonds:**

   - Diamond shape for conditions
   - Yellow background
   - Bold text

2. **Process Boxes:**

   - Rectangle with rounded corners
   - Light blue background

3. **Result Boxes:**

   - Rectangle with sharp corners
   - Green (4×4), Blue (8×8), Orange (16×16)
   - Include statistics

4. **Arrows:**
   - Labeled with conditions
   - "Yes"/"No" or ">100"/"<50"

### Caption:

"Figure 3.4: Adaptive Block Size Selection Flowchart. The algorithm analyzes content complexity using variance and gradient metrics to determine optimal block size for each region."

---

## Figure 3.5: Content-Aware Quantization Decision Tree

**Purpose:** Quantization matrix selection process  
**Location in Report:** Chapter 3, Page 17  
**Size:** 6 inches wide × 7 inches tall

### Decision Tree:

```
                    ┌─────────────────┐
                    │   Input Block   │
                    │     (8×8)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Calculate:     │
                    │  • Variance     │
                    │  • Edge Strength│
                    │  • Texture      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼────────┐    │    ┌────────▼─────────┐
    │  Variance > 100  │    │    │  Variance < 50   │
    │  (High Detail)   │    │    │    (Smooth)      │
    └─────────┬────────┘    │    └────────┬─────────┘
              │             │              │
    ┌─────────▼────────┐    │    ┌────────▼─────────┐
    │ Scale = 0.6      │    │    │ Scale = 1.3      │
    │ Preserve Detail  │    │    │ Compress More    │
    └─────────┬────────┘    │    └────────┬─────────┘
              │             │              │
              └──────────┬──┴──┬───────────┘
                         │     │
                ┌────────▼─────▼────────┐
                │ 50 ≤ Variance ≤ 100  │
                │   (Medium Detail)     │
                └────────┬──────────────┘
                         │
                ┌────────▼────────┐
                │  Scale = 0.7    │
                │   Balanced      │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │ Edge Detected?  │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │                 │
         ┌──────▼──────┐   ┌─────▼──────┐
         │ Edge = 0.8  │   │ Edge = 1.0 │
         │ (Preserve)  │   │  (Normal)  │
         └──────┬──────┘   └─────┬──────┘
                │                │
                └────────┬───────┘
                         │
                ┌────────▼────────┐
                │ Apply Perceptual│
                │   Weighting     │
                │  (HVS Model)    │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │ Final Q Matrix  │
                │ Q = Base × Scale│
                │   × Perceptual  │
                │   × Edge Factor │
                └─────────────────┘
```

### Styling:

1. **Color Coding:**

   - High detail path: Red tint
   - Medium detail path: Yellow tint
   - Low detail path: Green tint

2. **Box Styles:**

   - Input/Output: Rounded, blue
   - Calculations: Rectangle, gray
   - Decisions: Diamond, yellow
   - Results: Rounded, green

3. **Annotations:**
   - Add example values in small text
   - Show formula in final box

### Caption:

"Figure 3.5: Content-Aware Quantization Decision Tree. The algorithm adapts quantization matrices based on block variance, edge strength, and perceptual importance to optimize quality-compression tradeoff."

---

## Additional Figures for Chapter 5 (Results)

### Figure 5.1: Visual Comparison of Compression Results

**Layout:** 3×3 grid

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │   Paper     │  Improved   │
│   Image     │  Algorithm  │  Algorithm  │
├─────────────┼─────────────┼─────────────┤
│             │   Q = 30    │   Q = 30    │
│  (Reference)│ PSNR: 20.77 │ PSNR: 21.85 │
│             │ Grayscale   │ Full Color  │
├─────────────┼─────────────┼─────────────┤
│             │   Q = 50    │   Q = 50    │
│  (Reference)│ PSNR: 20.83 │ PSNR: 22.22 │
│             │ Grayscale   │ Full Color  │
├─────────────┼─────────────┼─────────────┤
│             │   Q = 80    │   Q = 80    │
│  (Reference)│ PSNR: 20.91 │ PSNR: 22.45 │
│             │ Grayscale   │ Full Color  │
└─────────────┴─────────────┴─────────────┘
```

**Use your actual generated images:**

- `sample_image.jpg` (original)
- `paper_result_q30.jpg`, `paper_result_q50.jpg`, `paper_result_q80.jpg`
- `improved_result_q30.jpg`, `improved_result_q50.jpg`, `improved_result_q80.jpg`

### Figure 5.2: PSNR Comparison Graph

**Type:** Line chart  
**X-axis:** Quality Level (30, 50, 80)  
**Y-axis:** PSNR (dB) (20-23 range)  
**Lines:**

- Blue line: Paper Algorithm
- Red line: Improved Algorithm
  **Data points:**
- Paper: (30, 20.77), (50, 20.83), (80, 20.91)
- Improved: (30, 21.85), (50, 22.22), (80, 22.45)

### Figure 5.3: Compression Ratio Comparison

**Type:** Grouped bar chart  
**X-axis:** Quality Level (30, 50, 80)  
**Y-axis:** Compression Ratio (0-60:1)  
**Bars:**

- Blue bars: Paper Algorithm
- Red bars: Improved Algorithm
  **Data:**
- Q30: Paper 41.32, Improved 51.58
- Q50: Paper 29.91, Improved 45.96
- Q80: Paper 17.05, Improved 40.55

### Figure 5.4: Difference Images

**Layout:** 2×3 grid showing error magnitudes

---

## Tools for Creating Diagrams:

1. **Microsoft Visio** - Professional diagrams
2. **Draw.io (diagrams.net)** - Free, web-based
3. **Lucidchart** - Online diagramming
4. **PowerPoint** - Simple flowcharts
5. **Python Matplotlib** - For graphs (Figures 5.2, 5.3)

---

## Summary Checklist:

- [ ] Figure 1.1: General Architecture
- [ ] Figure 3.1: Standard JPEG Pipeline
- [ ] Figure 3.2: Zigzag Pattern
- [ ] Figure 3.3: Proposed Architecture
- [ ] Figure 3.4: Adaptive Block Selection
- [ ] Figure 3.5: Quantization Decision Tree
- [ ] Figure 5.1: Visual Comparison (use generated images)
- [ ] Figure 5.2: PSNR Graph (create with matplotlib)
- [ ] Figure 5.3: Compression Ratio Chart (create with matplotlib)
- [ ] Figure 5.4: Difference Images (use generated images)

**All diagrams should be saved as high-resolution PNG or PDF for report insertion.**
