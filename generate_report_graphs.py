"""
Generate Graphs for Project Report
===================================

This script generates Figures 5.2 and 5.3 for the project report:
- Figure 5.2: PSNR Comparison Line Graph
- Figure 5.3: Compression Ratio Bar Chart

Run this script to automatically generate publication-quality graphs.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def generate_psnr_comparison():
    """Generate Figure 5.2: PSNR Comparison Graph"""
    
    # Data from our results
    quality_levels = [30, 50, 80]
    paper_psnr = [20.77, 20.83, 20.91]
    improved_psnr = [21.85, 22.22, 22.45]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot lines
    ax.plot(quality_levels, paper_psnr, 'o-', 
            color='#2196F3', linewidth=2, markersize=8,
            label='Paper Algorithm', markerfacecolor='white', 
            markeredgewidth=2)
    
    ax.plot(quality_levels, improved_psnr, 's-', 
            color='#F44336', linewidth=2, markersize=8,
            label='Improved Algorithm', markerfacecolor='white', 
            markeredgewidth=2)
    
    # Add value labels on points
    for i, (q, p_psnr, i_psnr) in enumerate(zip(quality_levels, paper_psnr, improved_psnr)):
        ax.text(q, p_psnr - 0.15, f'{p_psnr:.2f}', 
                ha='center', va='top', fontsize=8, color='#2196F3')
        ax.text(q, i_psnr + 0.15, f'{i_psnr:.2f}', 
                ha='center', va='bottom', fontsize=8, color='#F44336')
    
    # Formatting
    ax.set_xlabel('Quality Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('PSNR Comparison Across Quality Levels', 
                 fontsize=12, fontweight='bold', pad=15)
    
    ax.set_xticks(quality_levels)
    ax.set_ylim(20.5, 22.7)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    # Add improvement annotations
    for i, q in enumerate(quality_levels):
        improvement = improved_psnr[i] - paper_psnr[i]
        mid_y = (paper_psnr[i] + improved_psnr[i]) / 2
        ax.annotate(f'+{improvement:.2f} dB', 
                   xy=(q, mid_y), xytext=(q + 3, mid_y),
                   fontsize=8, color='green', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    plt.tight_layout()
    plt.savefig('Figure_5_2_PSNR_Comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: Figure_5_2_PSNR_Comparison.png")
    plt.close()


def generate_compression_ratio_chart():
    """Generate Figure 5.3: Compression Ratio Bar Chart"""
    
    # Data from our results
    quality_levels = ['Q30', 'Q50', 'Q80']
    paper_ratios = [41.32, 29.91, 17.05]
    improved_ratios = [51.58, 45.96, 40.55]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Set bar positions
    x = np.arange(len(quality_levels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, paper_ratios, width, 
                   label='Paper Algorithm',
                   color='#2196F3', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars2 = ax.bar(x + width/2, improved_ratios, width,
                   label='Improved Algorithm',
                   color='#F44336', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}:1',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement percentage labels
    for i, (paper, improved) in enumerate(zip(paper_ratios, improved_ratios)):
        improvement_pct = ((improved - paper) / paper) * 100
        ax.text(i, max(paper, improved) + 3, 
               f'+{improvement_pct:.0f}%',
               ha='center', va='bottom', fontsize=9, 
               color='green', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Quality Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Compression Ratio Comparison', 
                 fontsize=12, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(quality_levels)
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line for reference
    ax.axhline(y=30, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(2.5, 31, '30:1 baseline', fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig('Figure_5_3_Compression_Ratio.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: Figure_5_3_Compression_Ratio.png")
    plt.close()


def generate_combined_comparison():
    """Generate a combined comparison figure (bonus)"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: PSNR
    quality_levels = [30, 50, 80]
    paper_psnr = [20.77, 20.83, 20.91]
    improved_psnr = [21.85, 22.22, 22.45]
    
    ax1.plot(quality_levels, paper_psnr, 'o-', 
            color='#2196F3', linewidth=2, markersize=8,
            label='Paper Algorithm')
    ax1.plot(quality_levels, improved_psnr, 's-', 
            color='#F44336', linewidth=2, markersize=8,
            label='Improved Algorithm')
    
    ax1.set_xlabel('Quality Level', fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontweight='bold')
    ax1.set_title('(a) PSNR Comparison', fontweight='bold')
    ax1.set_xticks(quality_levels)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Compression Ratio
    quality_labels = ['Q30', 'Q50', 'Q80']
    paper_ratios = [41.32, 29.91, 17.05]
    improved_ratios = [51.58, 45.96, 40.55]
    
    x = np.arange(len(quality_labels))
    width = 0.35
    
    ax2.bar(x - width/2, paper_ratios, width, 
           label='Paper Algorithm', color='#2196F3', alpha=0.8)
    ax2.bar(x + width/2, improved_ratios, width,
           label='Improved Algorithm', color='#F44336', alpha=0.8)
    
    ax2.set_xlabel('Quality Level', fontweight='bold')
    ax2.set_ylabel('Compression Ratio', fontweight='bold')
    ax2.set_title('(b) Compression Ratio Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(quality_labels)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figure_Combined_Comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: Figure_Combined_Comparison.png (Bonus)")
    plt.close()


def generate_file_size_comparison():
    """Generate file size comparison chart (bonus)"""
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    quality_levels = ['Q30', 'Q50', 'Q80']
    paper_sizes = [17.7, 24.4, 42.9]  # KB
    improved_sizes = [14.2, 15.9, 18.0]  # KB
    
    x = np.arange(len(quality_levels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, paper_sizes, width,
                   label='Paper Algorithm', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, improved_sizes, width,
                   label='Improved Algorithm', color='#F44336', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} KB',
                   ha='center', va='bottom', fontsize=9)
    
    # Add savings percentage
    for i, (paper, improved) in enumerate(zip(paper_sizes, improved_sizes)):
        savings = ((paper - improved) / paper) * 100
        ax.text(i, max(paper, improved) + 2,
               f'{savings:.0f}% smaller',
               ha='center', va='bottom', fontsize=9,
               color='green', fontweight='bold')
    
    ax.set_xlabel('Quality Level', fontweight='bold')
    ax.set_ylabel('File Size (KB)', fontweight='bold')
    ax.set_title('File Size Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(quality_levels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figure_File_Size_Comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: Figure_File_Size_Comparison.png (Bonus)")
    plt.close()


def main():
    """Generate all report graphs"""
    print("Generating Report Graphs...")
    print("=" * 50)
    
    try:
        # Generate required figures
        generate_psnr_comparison()
        generate_compression_ratio_chart()
        
        # Generate bonus figures
        generate_combined_comparison()
        generate_file_size_comparison()
        
        print("=" * 50)
        print("✓ All graphs generated successfully!")
        print("\nGenerated Files:")
        print("  1. Figure_5_2_PSNR_Comparison.png")
        print("  2. Figure_5_3_Compression_Ratio.png")
        print("  3. Figure_Combined_Comparison.png (Bonus)")
        print("  4. Figure_File_Size_Comparison.png (Bonus)")
        print("\nThese images are ready to insert into your report!")
        
    except Exception as e:
        print(f"Error generating graphs: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    main()