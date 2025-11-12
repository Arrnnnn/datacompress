"""
Advanced JPEG Core Implementation - Part 1
==========================================

Core classes and methods for the improved JPEG algorithm.
"""

import numpy as np
import cv2
from collections import Counter, defaultdict
from scipy.fftpack import dct, idct
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import heapq
from typing import Tuple, Dict, List, Optional, Union
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import math

# Try to import numba for optimization
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class AdvancedJPEGCompressor:
    """
    Advanced JPEG Compressor implementing all improvements from new_improvements.md
    """
    
    def __init__(self, quality_factor: float = 0.8, enable_parallel: bool = True,
                 num_workers: Optional[int] = None):
        """
        Initialize the advanced JPEG compressor.
        
        Args:
            quality_factor: Quality factor (0.1 to 1.0)
            enable_parallel: Enable parallel processing
            num_workers: Number of worker processes
        """
        self.quality_factor = quality_factor
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers or mp.cpu_count()
        
        # Initialize components
        self._init_quantization_matrices()
        self._init_perceptual_models()
        
        # Thresholds from improvements.md enhanced
        self.variance_threshold_high = 100  # Enhanced from original 50
        self.variance_threshold_medium = 50  # Your original threshold
        self.gradient_threshold = 30
        
        print(f"Advanced JPEG Compressor initialized:")
        print(f"  Quality: {quality_factor}")
        print(f"  Parallel: {enable_parallel} ({self.num_workers} workers)")
        print(f"  Numba optimization: {NUMBA_AVAILABLE}")
    
    def _init_quantization_matrices(self):
        """Initialize enhanced quantization matrices."""
        # Base luminance matrix (from research paper)
        self.luma_quant_base = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Base chrominance matrix
        self.chroma_quant_base = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32)
        
        # Perceptual weighting matrices for different block sizes
        self.perceptual_weights_8x8 = np.array([
            [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
            [1.1, 1.2, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5],
            [1.2, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.5, 1.8, 2.0, 2.5, 3.5, 5.0, 6.0, 7.0],
            [2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 7.0, 8.0],
            [3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ])
        
        self.perceptual_weights_4x4 = self.perceptual_weights_8x8[:4, :4]
    
    def _init_perceptual_models(self):
        """Initialize perceptual optimization models."""
        # Contrast Sensitivity Function (CSF) matrix
        self.csf_matrix_8x8 = self._generate_csf_matrix(8)
        self.csf_matrix_4x4 = self._generate_csf_matrix(4)
    
    def _generate_csf_matrix(self, size: int) -> np.ndarray:
        """Generate Contrast Sensitivity Function matrix."""
        csf = np.ones((size, size))
        for u in range(size):
            for v in range(size):
                freq = np.sqrt(u*u + v*v) / size
                # CSF model - human eye sensitivity
                if freq > 0:
                    csf[u, v] = 1.0 / (1.0 + (freq * 10)**2)
                else:
                    csf[u, v] = 1.0
        return csf
    
    def calculate_block_complexity(self, block: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate comprehensive block complexity metrics.
        Combines variance approach from improvements.md with gradient analysis.
        
        Args:
            block: Image block
            
        Returns:
            Tuple of (variance, gradient_magnitude, total_complexity)
        """
        # Variance calculation (from improvements.md)
        variance = np.var(block)
        
        # Enhanced gradient analysis
        grad_x = np.gradient(block.astype(np.float32), axis=1)
        grad_y = np.gradient(block.astype(np.float32), axis=0)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Combined complexity metric
        total_complexity = variance + gradient_magnitude
        
        return variance, gradient_magnitude, total_complexity
    
    def determine_optimal_block_size(self, region: np.ndarray) -> int:
        """
        Determine optimal block size based on content analysis.
        Enhanced from improvements.md approach.
        
        Args:
            region: Image region to analyze
            
        Returns:
            Optimal block size (4, 8, or 16)
        """
        variance, gradient_mag, total_complexity = self.calculate_block_complexity(region)
        
        # Enhanced decision logic from improvements.md
        if total_complexity > self.variance_threshold_high:
            return 4   # High detail - smaller blocks
        elif total_complexity > self.variance_threshold_medium:
            return 8   # Medium complexity - standard blocks (your original threshold)
        else:
            return 16  # Low complexity - larger blocks
    
    def adaptive_quantization_matrix(self, block: np.ndarray, base_matrix: np.ndarray,
                                   is_luminance: bool = True) -> np.ndarray:
        """
        Generate content-aware quantization matrix.
        Enhanced implementation of improvements.md approach.
        
        Args:
            block: Image block
            base_matrix: Base quantization matrix
            is_luminance: Whether this is luminance channel
            
        Returns:
            Adaptive quantization matrix
        """
        variance, gradient_mag, total_complexity = self.calculate_block_complexity(block)
        
        # Adaptive scaling factors (enhanced from improvements.md)
        if variance > self.variance_threshold_high:
            scale_factor = 0.6  # High complexity - preserve more detail
        elif variance > self.variance_threshold_medium:
            scale_factor = 0.7  # Your original suggestion from improvements.md
        else:
            scale_factor = 1.3  # Low complexity - allow more compression
        
        # Edge preservation factor
        edge_strength = self._calculate_edge_strength(block)
        if edge_strength > 20:  # Strong edges detected
            edge_factor = 0.8  # Preserve edges better
        else:
            edge_factor = 1.0
        
        # Get appropriate perceptual weights
        block_size = block.shape[0]
        if block_size == 4:
            perceptual_weights = self.perceptual_weights_4x4
            csf_matrix = self.csf_matrix_4x4
        else:  # 8x8 or larger
            perceptual_weights = self.perceptual_weights_8x8
            csf_matrix = self.csf_matrix_8x8
        
        # Apply quality factor
        quality_scale = max(0.1, min(1.0, self.quality_factor))
        if quality_scale < 0.5:
            quality_multiplier = (0.5 / quality_scale)
        else:
            quality_multiplier = (2.0 - 2.0 * quality_scale)
        
        # Final adaptive matrix
        adaptive_matrix = (base_matrix * quality_multiplier * scale_factor * 
                          perceptual_weights * edge_factor)
        
        # Apply CSF weighting for perceptual optimization
        adaptive_matrix = adaptive_matrix / (csf_matrix + 0.1)
        
        # Ensure minimum values
        return np.maximum(adaptive_matrix, 1.0)
    
    def _calculate_edge_strength(self, block: np.ndarray) -> float:
        """Calculate edge strength in block."""
        # Sobel edge detection
        sobel_x = cv2.Sobel(block.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(block.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.mean(edge_magnitude)
    
    def enhanced_dct_processing(self, block: np.ndarray) -> np.ndarray:
        """
        Enhanced DCT with improved precision and adaptive transforms.
        
        Args:
            block: Input block
            
        Returns:
            DCT coefficients
        """
        block_size = block.shape[0]
        
        # High-precision DCT computation
        block_centered = block.astype(np.float64) - 128.0
        
        # Apply DCT based on block size
        if block_size <= 8:
            dct_coeffs = cv2.dct(block_centered.astype(np.float32)).astype(np.float64)
        else:
            # Use scipy for larger blocks
            dct_coeffs = dct(dct(block_centered.T, norm='ortho').T, norm='ortho')
        
        # Adaptive coefficient thresholding
        threshold = self._calculate_adaptive_threshold(block_centered)
        dct_coeffs = self._soft_threshold(dct_coeffs, threshold)
        
        return dct_coeffs
    
    def _calculate_adaptive_threshold(self, block: np.ndarray) -> float:
        """Calculate adaptive threshold for coefficient thresholding."""
        noise_variance = np.var(block) * 0.01  # Estimate noise
        return np.sqrt(2 * noise_variance * np.log(block.size))
    
    def _soft_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding to coefficients."""
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def intelligent_chroma_subsampling(self, cb_channel: np.ndarray, 
                                     cr_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Intelligent chroma subsampling based on content analysis.
        
        Args:
            cb_channel: Cb channel
            cr_channel: Cr channel
            
        Returns:
            Subsampled channels and subsampling ratio used
        """
        # Analyze chroma importance
        cb_variance = np.var(cb_channel)
        cr_variance = np.var(cr_channel)
        chroma_variance = cb_variance + cr_variance
        
        # Calculate color complexity
        color_complexity = self._calculate_color_complexity(cb_channel, cr_channel)
        
        # Apply anti-aliasing filter before subsampling
        cb_filtered = gaussian_filter(cb_channel.astype(np.float32), sigma=0.5)
        cr_filtered = gaussian_filter(cr_channel.astype(np.float32), sigma=0.5)
        
        # Adaptive subsampling decision
        if color_complexity > 80:  # High color detail
            subsampling_ratio = "4:2:2"
            cb_sub = cb_filtered[::1, ::2]  # Less aggressive
            cr_sub = cr_filtered[::1, ::2]
        elif color_complexity > 40:  # Medium color detail
            subsampling_ratio = "4:2:0"  # Standard
            cb_sub = cb_filtered[::2, ::2]
            cr_sub = cr_filtered[::2, ::2]
        else:  # Low color detail
            subsampling_ratio = "4:1:1"  # Aggressive
            cb_sub = cb_filtered[::2, ::4]
            cr_sub = cr_filtered[::2, ::4]
        
        return cb_sub.astype(np.uint8), cr_sub.astype(np.uint8), subsampling_ratio
    
    def _calculate_color_complexity(self, cb_channel: np.ndarray, cr_channel: np.ndarray) -> float:
        """Calculate color complexity metric."""
        # Gradient-based color complexity
        cb_grad = np.gradient(cb_channel.astype(np.float32))
        cr_grad = np.gradient(cr_channel.astype(np.float32))
        
        cb_complexity = np.mean(np.abs(cb_grad[0])) + np.mean(np.abs(cb_grad[1]))
        cr_complexity = np.mean(np.abs(cr_grad[0])) + np.mean(np.abs(cr_grad[1]))
        
        return cb_complexity + cr_complexity