"""
Create Side-by-Side Comparison Images
Shows Original, Standard JPEG, and Improved Algorithm results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_comparison_image(original_path, standard_path, improved_path, output_path, quality_level):
    """
    Create a side-by-side comparison image with labels and metrics
    """
    # Read images
    original = cv2.imread(original_path)
    standard = cv2.imread(standard_path)
    improved = cv2.imread(improved_path)
    
    if original is None or standard is None or improved is None:
        print(f"Error: Could not load one or more images")
        print(f"Original: {original_path}")
        print(f"Standard: {standard_path}")
        print(f"Improved: {improved_path}")
        return False
    
    # Convert BGR to RGB for matplotlib
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    standard = cv2.cvtColor(standard, cv2.COLOR_BGR2RGB)
    improved = cv2.cvtColor(improved, cv2.COLOR_BGR2RGB)
    
    # Calculate PSNR
    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    psnr_standard = calculate_psnr(original, standard)
    psnr_improved = calculate_psnr(original, improved)
    
    # Get file sizes
    import os
    size_original = os.path.getsize(original_path) / 1024  # KB
    size_standard = os.path.getsize(standard_path) / 1024
    size_improved = os.path.getsize(improved_path) / 1024
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'JPEG Compression Comparison (Quality {quality_level})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Original Image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    axes[0].text(0.5, -0.08, f'Size: {size_original:.1f} KB\nReference Quality', 
                ha='center', va='top', transform=axes[0].transAxes, 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[0].axis('off')
    
    # Standard JPEG
    axes[1].imshow(standard)
    axes[1].set_title('Standard JPEG', fontsize=14, fontweight='bold', pad=10, color='red')
    axes[1].text(0.5, -0.08, 
                f'Size: {size_standard:.1f} KB\nPSNR: {psnr_standard:.2f} dB\nGrayscale Output', 
                ha='center', va='top', transform=axes[1].transAxes, 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    axes[1].axis('off')
    
    # Improved Algorithm
    axes[2].imshow(improved)
    axes[2].set_title('Improved Algorithm', fontsize=14, fontweight='bold', pad=10, color='green')
    improvement_text = f'Size: {size_improved:.1f} KB ({(size_standard-size_improved)/size_standard*100:.1f}% smaller)\n'
    improvement_text += f'PSNR: {psnr_improved:.2f} dB (+{psnr_improved-psnr_standard:.2f} dB)\n'
    improvement_text += f'Full Color Output'
    axes[2].text(0.5, -0.08, improvement_text,
                ha='center', va='top', transform=axes[2].transAxes, 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Original (Reference)'),
        mpatches.Patch(color='#ffcccc', label='Standard JPEG (Baseline)'),
        mpatches.Patch(color='#ccffcc', label='Improved Algorithm (Ours)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison image: {output_path}")
    plt.close()
    
    return True


def create_zoomed_comparison(original_path, standard_path, improved_path, output_path, 
                            crop_region, quality_level):
    """
    Create a zoomed-in comparison showing detail preservation
    """
    # Read images
    original = cv2.imread(original_path)
    standard = cv2.imread(standard_path)
    improved = cv2.imread(improved_path)
    
    if original is None or standard is None or improved is None:
        print(f"Error: Could not load images for zoomed comparison")
        return False
    
    # Convert BGR to RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    standard = cv2.cvtColor(standard, cv2.COLOR_BGR2RGB)
    improved = cv2.cvtColor(improved, cv2.COLOR_BGR2RGB)
    
    # Extract crop region
    y1, y2, x1, x2 = crop_region
    crop_orig = original[y1:y2, x1:x2]
    crop_std = standard[y1:y2, x1:x2]
    crop_imp = improved[y1:y2, x1:x2]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Detail Comparison - Zoomed View (Quality {quality_level})', 
                 fontsize=16, fontweight='bold')
    
    # Top row - Full images with crop box
    for idx, (img, title, ax) in enumerate(zip(
        [original, standard, improved],
        ['Original', 'Standard JPEG', 'Improved Algorithm'],
        axes[0]
    )):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        # Draw crop box
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')
    
    # Bottom row - Zoomed crops
    for idx, (crop, title, ax, color) in enumerate(zip(
        [crop_orig, crop_std, crop_imp],
        ['Original (Detail)', 'Standard JPEG (Detail)', 'Improved Algorithm (Detail)'],
        axes[1],
        ['blue', 'red', 'green']
    )):
        ax.imshow(crop)
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved zoomed comparison: {output_path}")
    plt.close()
    
    return True


def create_all_comparisons():
    """
    Create all comparison images for different quality levels
    """
    print("Creating Comparison Images...")
    print("=" * 60)
    
    # Quality levels to compare
    qualities = [30, 50, 80]
    
    for quality in qualities:
        print(f"\nProcessing Quality {quality}...")
        
        # File paths
        original = 'sample_image.jpg'
        standard = f'improved_jpeg_q{quality}_standard.jpg'
        improved = f'improved_jpeg_q{quality}_adaptive.jpg'
        output = f'comparison_q{quality}.jpg'
        
        # Create side-by-side comparison
        success = create_comparison_image(original, standard, improved, output, quality)
        
        if success and quality == 50:
            # Create zoomed comparison for Q50
            # Adjust crop region based on your image
            height, width = cv2.imread(original).shape[:2]
            crop_region = (
                height // 4,           # y1
                height // 4 + 100,     # y2
                width // 4,            # x1
                width // 4 + 100       # x2
            )
            zoomed_output = f'comparison_q{quality}_zoomed.jpg'
            create_zoomed_comparison(original, standard, improved, zoomed_output, 
                                   crop_region, quality)
    
    print("\n" + "=" * 60)
    print("✓ All comparison images created successfully!")
    print("\nGenerated files:")
    print("  - comparison_q30.jpg")
    print("  - comparison_q50.jpg")
    print("  - comparison_q50_zoomed.jpg")
    print("  - comparison_q80.jpg")


if __name__ == "__main__":
    create_all_comparisons()
