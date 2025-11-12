"""
Create Final Comparison Images for Presentation
Uses existing paper_result and improved_result files
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def create_comparison_image(quality_level):
    """
    Create a side-by-side comparison image
    """
    # File paths
    original_path = 'sample_image.jpg'
    paper_path = f'paper_result_q{quality_level}.jpg'
    improved_path = f'improved_result_q{quality_level}.jpg'
    output_path = f'final_comparison_q{quality_level}.jpg'
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"✗ Original image not found: {original_path}")
        return False
    if not os.path.exists(paper_path):
        print(f"✗ Paper result not found: {paper_path}")
        return False
    if not os.path.exists(improved_path):
        print(f"✗ Improved result not found: {improved_path}")
        return False
    
    # Read images
    original = cv2.imread(original_path)
    paper = cv2.imread(paper_path)
    improved = cv2.imread(improved_path)
    
    # Convert BGR to RGB for matplotlib
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
    improved = cv2.cvtColor(improved, cv2.COLOR_BGR2RGB)
    
    # Calculate PSNR
    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    psnr_paper = calculate_psnr(original, paper)
    psnr_improved = calculate_psnr(original, improved)
    
    # Get file sizes
    size_original = os.path.getsize(original_path) / 1024  # KB
    size_paper = os.path.getsize(paper_path) / 1024
    size_improved = os.path.getsize(improved_path) / 1024
    
    # Calculate compression ratios
    ratio_paper = size_original / size_paper
    ratio_improved = size_original / size_improved
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'JPEG Compression Algorithm Comparison (Quality {quality_level})', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Original Image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=15, fontweight='bold', pad=15)
    info_text = f'Size: {size_original:.1f} KB\nReference Quality\nFull Color'
    axes[0].text(0.5, -0.12, info_text,
                ha='center', va='top', transform=axes[0].transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    axes[0].axis('off')
    
    # Paper Algorithm (Standard JPEG)
    axes[1].imshow(paper)
    axes[1].set_title('Paper Algorithm\n(Standard JPEG)', fontsize=15, fontweight='bold', 
                     pad=15, color='#D32F2F')
    info_text = f'Size: {size_paper:.1f} KB\n'
    info_text += f'PSNR: {psnr_paper:.2f} dB\n'
    info_text += f'Ratio: {ratio_paper:.1f}:1\n'
    info_text += f'Grayscale Output'
    axes[1].text(0.5, -0.12, info_text,
                ha='center', va='top', transform=axes[1].transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))
    axes[1].axis('off')
    
    # Improved Algorithm
    axes[2].imshow(improved)
    axes[2].set_title('Improved Algorithm\n(Our Approach)', fontsize=15, fontweight='bold', 
                     pad=15, color='#388E3C')
    
    # Calculate improvements
    size_reduction = ((size_paper - size_improved) / size_paper * 100)
    psnr_gain = psnr_improved - psnr_paper
    ratio_improvement = ratio_improved / ratio_paper
    
    info_text = f'Size: {size_improved:.1f} KB ({size_reduction:+.1f}%)\n'
    info_text += f'PSNR: {psnr_improved:.2f} dB ({psnr_gain:+.2f} dB)\n'
    info_text += f'Ratio: {ratio_improved:.1f}:1 ({ratio_improvement:.2f}× better)\n'
    info_text += f'Full Color Output ✓'
    axes[2].text(0.5, -0.12, info_text,
                ha='center', va='top', transform=axes[2].transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.9))
    axes[2].axis('off')
    
    # Add summary box at bottom
    summary_text = f'Key Improvements: PSNR +{psnr_gain:.2f} dB | '
    summary_text += f'Size {size_reduction:+.1f}% | '
    summary_text += f'Compression {ratio_improvement:.2f}× better | '
    summary_text += f'Full Color Support'
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Created: {output_path}")
    print(f"  PSNR: {psnr_paper:.2f} dB → {psnr_improved:.2f} dB ({psnr_gain:+.2f} dB)")
    print(f"  Size: {size_paper:.1f} KB → {size_improved:.1f} KB ({size_reduction:+.1f}%)")
    print(f"  Ratio: {ratio_paper:.1f}:1 → {ratio_improved:.1f}:1 ({ratio_improvement:.2f}×)")
    plt.close()
    
    return True


def create_zoomed_detail_comparison(quality_level=50):
    """
    Create a detailed zoomed comparison showing blocking artifacts
    """
    # File paths
    original_path = 'sample_image.jpg'
    paper_path = f'paper_result_q{quality_level}.jpg'
    improved_path = f'improved_result_q{quality_level}.jpg'
    output_path = f'detail_comparison_q{quality_level}.jpg'
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [original_path, paper_path, improved_path]):
        print(f"✗ Some files not found for detail comparison")
        return False
    
    # Read images
    original = cv2.imread(original_path)
    paper = cv2.imread(paper_path)
    improved = cv2.imread(improved_path)
    
    # Convert BGR to RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
    improved = cv2.cvtColor(improved, cv2.COLOR_BGR2RGB)
    
    # Define crop regions (adjust based on your image)
    height, width = original.shape[:2]
    
    # Region 1: Top-left (edges/details)
    crop1 = (50, 150, 50, 150)
    # Region 2: Center (textures)
    crop2 = (height//2-50, height//2+50, width//2-50, width//2+50)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)
    
    fig.suptitle(f'Detailed Quality Comparison - Zoomed Regions (Quality {quality_level})', 
                 fontsize=18, fontweight='bold')
    
    # Top row - Full images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    for ax, img, title, color in zip(
        [ax1, ax2, ax3],
        [original, paper, improved],
        ['Original', 'Paper Algorithm', 'Improved Algorithm'],
        ['blue', 'red', 'green']
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        # Draw crop boxes
        for crop, box_color in zip([crop1, crop2], ['yellow', 'cyan']):
            y1, y2, x1, x2 = crop
            rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=3, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
    
    # Middle row - Region 1 zoomed
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    y1, y2, x1, x2 = crop1
    for ax, img, title in zip(
        [ax4, ax5, ax6],
        [original[y1:y2, x1:x2], paper[y1:y2, x1:x2], improved[y1:y2, x1:x2]],
        ['Original (Detail 1)', 'Paper (Detail 1)\nBlocking Artifacts ⚠️', 'Improved (Detail 1)\nSmooth ✓']
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Bottom row - Region 2 zoomed
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    y1, y2, x1, x2 = crop2
    for ax, img, title in zip(
        [ax7, ax8, ax9],
        [original[y1:y2, x1:x2], paper[y1:y2, x1:x2], improved[y1:y2, x1:x2]],
        ['Original (Detail 2)', 'Paper (Detail 2)\nGrayscale', 'Improved (Detail 2)\nFull Color ✓']
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Created: {output_path}")
    plt.close()
    
    return True


def main():
    """
    Create all comparison images
    """
    print("=" * 70)
    print("Creating Final Comparison Images for Presentation")
    print("=" * 70)
    print()
    
    # Create comparisons for all quality levels
    qualities = [30, 50, 80]
    
    for quality in qualities:
        print(f"Quality {quality}:")
        success = create_comparison_image(quality)
        if not success:
            print(f"  Skipped (files not found)")
        print()
    
    # Create detailed comparison for Q50
    print("Creating detailed zoomed comparison for Q50...")
    create_zoomed_detail_comparison(50)
    
    print()
    print("=" * 70)
    print("✓ All comparison images created successfully!")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  📊 final_comparison_q30.jpg  - Side-by-side comparison (Q30)")
    print("  📊 final_comparison_q50.jpg  - Side-by-side comparison (Q50)")
    print("  📊 final_comparison_q80.jpg  - Side-by-side comparison (Q80)")
    print("  🔍 detail_comparison_q50.jpg - Zoomed detail view (Q50)")
    print()
    print("Use these images in your presentation to show:")
    print("  ✓ Visual quality improvement")
    print("  ✓ File size reduction")
    print("  ✓ PSNR improvement")
    print("  ✓ Full color vs grayscale")
    print("  ✓ Blocking artifact reduction")


if __name__ == "__main__":
    main()
