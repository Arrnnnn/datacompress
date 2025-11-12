# Troubleshooting Guide - Common Errors and Fixes

## 🔧 Common Errors and Solutions

---

## Error 1: ModuleNotFoundError: No module named 'cv2'

**Error Message:**

```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**

```bash
pip install opencv-python
```

**If that doesn't work:**

```bash
pip install opencv-python --upgrade
```

**For Python 3:**

```bash
pip3 install opencv-python
```

---

## Error 2: ModuleNotFoundError: No module named 'numpy'

**Error Message:**

```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:**

```bash
pip install numpy
```

**If that doesn't work:**

```bash
python -m pip install numpy
```

---

## Error 3: ModuleNotFoundError: No module named 'scipy'

**Error Message:**

```
ModuleNotFoundError: No module named 'scipy'
```

**Solution:**

```bash
pip install scipy
```

---

## Error 4: ModuleNotFoundError: No module named 'matplotlib'

**Error Message:**

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:**

```bash
pip install matplotlib
```

---

## Error 5: Python is not recognized

**Error Message:**

```
'python' is not recognized as an internal or external command
```

**Solution 1 - Try python3:**

```bash
python3 improved_jpeg_complete.py
```

**Solution 2 - Add Python to PATH (Windows):**

1. Search "Environment Variables" in Windows
2. Edit "Path" variable
3. Add Python installation directory
4. Restart terminal

**Solution 3 - Use full path:**

```bash
C:\Python39\python.exe improved_jpeg_complete.py
```

---

## Error 6: Permission Denied

**Error Message:**

```
PermissionError: [Errno 13] Permission denied
```

**Solution (Windows):**

```bash
pip install --user numpy opencv-python scipy matplotlib
```

**Solution (Mac/Linux):**

```bash
sudo pip install numpy opencv-python scipy matplotlib
```

Or use pip3:

```bash
pip3 install --user numpy opencv-python scipy matplotlib
```

---

## Error 7: File Not Found

**Error Message:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'improved_jpeg_complete.py'
```

**Solution:**
Make sure you're in the correct directory:

```bash
# Check current directory
pwd  # Mac/Linux
cd   # Windows

# List files
ls   # Mac/Linux
dir  # Windows

# Navigate to project directory
cd path/to/your/project
```

---

## Error 8: Image File Not Found

**Error Message:**

```
error: OpenCV(4.x.x) :-1: error: (-5:Bad argument) in function 'imread'
```

**Solution:**
The program will create a test image automatically. Just let it run.

Or create your own:

```python
# The program will handle this automatically
# No action needed
```

---

## Error 9: Memory Error

**Error Message:**

```
MemoryError: Unable to allocate array
```

**Solution:**
Use a smaller image:

```python
# Resize image before processing
import cv2
img = cv2.imread('large_image.jpg')
img = cv2.resize(img, (512, 512))
cv2.imwrite('sample_image.jpg', img)
```

---

## Error 10: Syntax Error

**Error Message:**

```
SyntaxError: invalid syntax
```

**Solution:**
Make sure you're using Python 3.7 or higher:

```bash
python --version
```

If version is too old, install newer Python from python.org

---

## Error 11: Import Error with specific function

**Error Message:**

```
ImportError: cannot import name 'xxx' from 'module'
```

**Solution:**
Update all packages:

```bash
pip install --upgrade numpy opencv-python scipy matplotlib
```

---

## Error 12: DLL Load Failed (Windows)

**Error Message:**

```
ImportError: DLL load failed while importing cv2
```

**Solution:**
Install Visual C++ Redistributable:

1. Download from Microsoft website
2. Install both x86 and x64 versions
3. Restart computer

Or reinstall opencv:

```bash
pip uninstall opencv-python
pip install opencv-python
```

---

## Error 13: Tkinter Error

**Error Message:**

```
ImportError: No module named '_tkinter'
```

**Solution:**
This is for matplotlib display. Install tkinter:

**Ubuntu/Debian:**

```bash
sudo apt-get install python3-tk
```

**Mac:**

```bash
brew install python-tk
```

**Windows:**
Reinstall Python with tkinter option checked

---

## Error 14: Slow Performance / Hanging

**Symptoms:**
Program runs but takes very long or appears frozen

**Solution:**
This is normal for large images. Wait for completion.

To speed up:

1. Use smaller images (512×512 recommended)
2. Reduce quality levels tested
3. Disable parallel processing if causing issues

---

## Error 15: No Output Files Generated

**Symptoms:**
Program runs successfully but no output images

**Solution:**
Check current directory:

```bash
# Windows
dir improved_jpeg_*

# Mac/Linux
ls improved_jpeg_*
```

Files should be in the same directory as the script.

---

## 🔍 Diagnostic Commands

### Check Python Installation:

```bash
python --version
# Should show: Python 3.7.x or higher
```

### Check pip Installation:

```bash
pip --version
# Should show pip version
```

### Check All Required Packages:

```bash
pip list | grep -E "numpy|opencv|scipy|matplotlib"
```

### Test Individual Imports:

```bash
python -c "import numpy; print('NumPy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import scipy; print('SciPy OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

---

## 🆘 Still Having Issues?

### Complete Reinstall:

```bash
# Uninstall all packages
pip uninstall numpy opencv-python scipy matplotlib -y

# Clear pip cache
pip cache purge

# Reinstall everything
pip install numpy opencv-python scipy matplotlib
```

### Use Virtual Environment:

```bash
# Create virtual environment
python -m venv jpeg_env

# Activate it
# Windows:
jpeg_env\Scripts\activate
# Mac/Linux:
source jpeg_env/bin/activate

# Install packages
pip install numpy opencv-python scipy matplotlib

# Run program
python improved_jpeg_complete.py
```

---

## 📋 Pre-Run Checklist

Before running the program, verify:

- [ ] Python 3.7+ installed (`python --version`)
- [ ] pip working (`pip --version`)
- [ ] NumPy installed (`pip show numpy`)
- [ ] OpenCV installed (`pip show opencv-python`)
- [ ] SciPy installed (`pip show scipy`)
- [ ] Matplotlib installed (`pip show matplotlib`)
- [ ] In correct directory (`ls` or `dir` shows .py file)
- [ ] Have write permissions in directory

---

## 💡 Quick Fix - Install Everything at Once

```bash
# One command to install all dependencies
pip install numpy opencv-python scipy matplotlib

# If that fails, try:
python -m pip install numpy opencv-python scipy matplotlib

# If still fails, try with --user:
pip install --user numpy opencv-python scipy matplotlib

# For Python 3 specifically:
pip3 install numpy opencv-python scipy matplotlib
```

---

## 🎯 Minimal Test Script

Create a file called `test_imports.py`:

```python
print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print("✗ NumPy import failed:", e)

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print("✗ OpenCV import failed:", e)

try:
    import scipy
    print("✓ SciPy imported successfully")
except ImportError as e:
    print("✗ SciPy import failed:", e)

try:
    import matplotlib
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print("✗ Matplotlib import failed:", e)

print("\nAll imports successful! Ready to run main program.")
```

Run it:

```bash
python test_imports.py
```

If all show ✓, you're ready to run the main program!

---

## 📞 Getting Help

If none of these solutions work:

1. **Copy the exact error message**
2. **Note your Python version** (`python --version`)
3. **Note your OS** (Windows/Mac/Linux)
4. **Share the error with me**

I'll help you fix it! 🚀
