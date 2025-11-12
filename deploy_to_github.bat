@echo off
echo ========================================
echo GitHub Deployment Script
echo ========================================
echo.

echo Step 1: Initializing Git repository...
git init
echo.

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Creating initial commit...
git commit -m "Initial commit: Improved JPEG compression algorithm with adaptive block processing"
echo.

echo Step 4: Renaming branch to main...
git branch -M main
echo.

echo ========================================
echo IMPORTANT: Next Steps
echo ========================================
echo.
echo 1. Create a repository on GitHub:
echo    - Go to https://github.com/new
echo    - Name: improved-jpeg-compression
echo    - Do NOT initialize with README
echo.
echo 2. Copy your repository URL
echo    Example: https://github.com/YOUR_USERNAME/improved-jpeg-compression.git
echo.
echo 3. Run these commands (replace with your URL):
echo    git remote add origin YOUR_REPOSITORY_URL
echo    git push -u origin main
echo.
echo ========================================
pause
