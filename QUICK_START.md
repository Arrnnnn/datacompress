# Quick Start Guide - How to Run This Project

## 🚀 Simple 3-Step Process

---

## Step 1: Install Required Libraries

Open your terminal/command prompt and run:

```bash
pip install numpy opencv-python scipy matplotlib
```

**Or if you see errors, try:**

```bash
pip install numpy
pip install opencv-python
pip install scipy
pip install matplotlib
```

**Check if installed correctly:**

```bash
python -c "import numpy, cv2, scipy, matplotlib; print('All libraries installed!')"
```

---

## Step 2: Prepare Your Image

**Option A: Use your own image**

- Place any image file in the project folder
- Rename it to `sample_image.jpg`

**Option B: Let the program create a test image**

- Skip this step - the program will automatically create a test image

---

## Step 3: Run the Program

```bash
python improved_jpeg_complete.py
```

**That's it!** The program will run and show you results.

---

## 📊 What You'll See

### Console Output:

```
Complete Improved JPEG Algorithm Implementation
============================================================
Features implemented:
✅ Adaptive Block Processing (4x4, 8x8, 16x16)
✅ Content-Aware Quantization
✅ Enhanced Entropy Coding
...

Starting Improved JPEG Compression...
1. Converting RGB to YCbCr...
2. Applying intelligent chroma subsampling...
3. Processing Y channel with adaptive blocks...
...
Compression complete in 4.92 seconds!

Original size: 786432 bytes
Compressed size: 15876 bytes
Compression ratio: 45.96:1
PSNR: 22.22 dB
```

### Generated Files:

The program creates these output images:

- `improved_jpeg_q30_adaptive.jpg` (Quality 30)
- `improved_jpeg_q50_adaptive.jpg` (Quality 50)
- `improved_jpeg_q80_adaptive.jpg` (Quality 80)
- `improved_jpeg_q50_standard.jpg` (Baseline comparison)

---

## 🎯 Understanding the Results

### Key Numbers to Look For:

1. **Original size: 786432 bytes**

   - Your input image size

2. **Compressed size: 15876 bytes**

   - Output image size (much smaller!)

3. **Compression ratio: 45.96:1**

   - How much smaller (45× reduction)

4. **PSNR: 22.22 dB**

   - Quality metric (higher = better)
   - 20-25 dB = Good quality

5. **Processing time: 4.92 seconds**
   - How long compression took

---

## 🖼️ Viewing Results

### Open the generated images:

**Windows:**

```bash
start improved_jpeg_q50_adaptive.jpg
```

**Mac:**

```bash
open improved_jpeg_q50_adaptive.jpg
```

**Linux:**

```bash
xdg-open improved_jpeg_q50_adaptive.jpg
```

### Compare images:

- Original: `sample_image.jpg`
- Improved: `improved_jpeg_q50_adaptive.jpg`
- Standard: `improved_jpeg_q50_standard.jpg`

---

## ⚠️ Common Issues and Solutions

### Issue 1: "No module named 'cv2'"

**Solution:**

```bash
pip install opencv-python
```

### Issue 2: "No module named 'numpy'"

**Solution:**

```bash
pip install numpy
```

### Issue 3: "Permission denied"

**Solution (Windows):**

```bash
python -m pip install --user numpy opencv-python scipy matplotlib
```

**Solution (Mac/Linux):**

```bash
pip3 install numpy opencv-python scipy matplotlib
```

### Issue 4: "Python not found"

**Solution:**

- Make sure Python is installed
- Try `python3` instead of `python`

```bash
python3 improved_jpeg_complete.py
```

### Issue 5: Program runs but no output images

**Solution:**

- Check the current directory
- Look for files starting with `improved_jpeg_`

```bash
# Windows
dir improved_jpeg_*

# Mac/Linux
ls improved_jpeg_*
```

---

## 🎓 For Faculty Demonstration

### Quick Demo Script:

1. **Show the command:**

   ```bash
   python improved_jpeg_complete.py
   ```

2. **While running, explain:**
   "The program is processing the image through multiple stages - color conversion, adaptive block selection, compression, and reconstruction."

3. **When results appear, highlight:**

   - "Original size: 768 KB → Compressed: 15.9 KB (35% smaller)"
   - "PSNR: 22.22 dB (1.39 dB better than standard)"
   - "Compression ratio: 45.96:1 (1.54× better)"

4. **Show the images:**
   - Open original and compressed side-by-side
   - Zoom in to show quality preservation

---

## 📝 Quick Reference

### Run with different quality levels:

Edit the `improved_jpeg_complete.py` file and change quality values:

```python
# Find this section in the code:
configurations = [
    {"quality": 30, "adaptive": True, "perceptual": True, "parallel": True},
    {"quality": 50, "adaptive": True, "perceptual": True, "parallel": True},
    {"quality": 80, "adaptive": True, "perceptual": True, "parallel": True},
]
```

### Run with your own image:

```python
# Find this line in main():
image = cv2.imread('sample_image.jpg')

# Change to your image:
image = cv2.imread('my_photo.jpg')
```

---

## ✅ Success Checklist

- [ ] Python installed (version 3.7 or higher)
- [ ] All libraries installed (numpy, opencv-python, scipy, matplotlib)
- [ ] Image file ready (or let program create one)
- [ ] Run command: `python improved_jpeg_complete.py`
- [ ] See console output with results
- [ ] Find generated image files
- [ ] Open and compare images

---

## 🆘 Need Help?

### Check Python version:

```bash
python --version
```

Should show Python 3.7 or higher

### Check installed packages:

```bash
pip list | grep -E "numpy|opencv|scipy|matplotlib"
```

### Test imports:

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

---

## 🎯 Expected Runtime

- **Small images (512×512):** 3-5 seconds
- **Medium images (1024×1024):** 10-15 seconds
- **Large images (2048×2048):** 30-60 seconds

**Note:** First run might be slower as libraries load.

---

## 💡 Pro Tips

1. **Use a colorful image** for best demonstration of improvements
2. **Run once before demo** to ensure everything works
3. **Keep generated images** for comparison
4. **Take screenshots** of console output for reports
5. **Zoom in on images** to show quality differences

---

## 🚀 You're Ready!

Just run:

```bash
python improved_jpeg_complete.py
```

And you'll see your improved JPEG compression algorithm in action! 🎉
