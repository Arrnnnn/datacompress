"""
Quantization component for the compression pipeline.

This module implements quantization and dequantization operations with configurable
quality parameters for controlling compression vs quality tradeoffs.
"""

import numpy as np
from typing import Optional


class Quantizer:
    """Manages quantization and inverse quantization operations."""
    
    def __init__(self, quality: float = 0.8, block_size: int = 8):
        """
        Initialize quantizer with quality parameter.
        
        Args:
            quality: Quality factor between 0.1 and 1.0 (higher = better quality)
            block_size: Size of blocks for quantization table generation
            
        Raises:
            ValueError: If quality is not in valid range
        """
        if not 0.1 <= quality <= 1.0:
            raise ValueError("quality must be between 0.1 and 1.0")
        
        self.quality = quality
        self.block_size = block_size
        self.quantization_table = self._generate_quantization_table(quality)
    
    def quantize(self, dct_coefficients: np.ndarray) -> np.ndarray:
        """
        Apply quantization to DCT coefficients.
        
        Args:
            dct_coefficients: 2D array of DCT coefficients
            
        Returns:
            Quantized coefficients as integer array
            
        Raises:
            ValueError: If coefficient array shape doesn't match block size
        """
        if dct_coefficients.shape != (self.block_size, self.block_size):
            raise ValueError(f"DCT coefficients must be {self.block_size}x{self.block_size}")
        
        # Apply quantization: divide by quantization table and round
        quantized = np.round(dct_coefficients / self.quantization_table)
        
        # Convert to integers and clip to reasonable range to prevent overflow
        quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
        
        return quantized
    
    def dequantize(self, quantized_coeffs: np.ndarray) -> np.ndarray:
        """
        Reverse quantization process to restore frequency coefficients.
        
        Args:
            quantized_coeffs: 2D array of quantized coefficients
            
        Returns:
            Dequantized coefficients as float array
            
        Raises:
            ValueError: If coefficient array shape doesn't match block size
        """
        if quantized_coeffs.shape != (self.block_size, self.block_size):
            raise ValueError(f"Quantized coefficients must be {self.block_size}x{self.block_size}")
        
        # Reverse quantization: multiply by quantization table
        dequantized = quantized_coeffs.astype(np.float32) * self.quantization_table
        
        return dequantized
    
    def _generate_quantization_table(self, quality: float) -> np.ndarray:
        """
        Generate quantization table based on quality parameter.
        
        This creates a quantization table similar to JPEG's approach, where
        higher frequency components are quantized more aggressively.
        
        Args:
            quality: Quality factor between 0.1 and 1.0
            
        Returns:
            Quantization table as 2D numpy array
        """
        # Base quantization table (similar to JPEG luminance table)
        base_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Adjust table size if different block size is used
        if self.block_size != 8:
            # For non-8x8 blocks, create a scaled version
            base_table = self._scale_quantization_table(base_table, self.block_size)
        
        # Scale table based on quality
        if quality >= 0.5:
            # High quality: scale down the quantization values
            scale_factor = (1.0 - quality) * 2.0  # 0 to 1 for quality 0.5 to 1.0
        else:
            # Low quality: scale up the quantization values more aggressively
            scale_factor = 2.0 - quality * 4.0  # 2 to 0 for quality 0 to 0.5
        
        # Apply scaling with minimum value of 1 to avoid division by zero
        scaled_table = np.maximum(base_table * scale_factor, 1.0)
        
        return scaled_table.astype(np.float32)
    
    def _scale_quantization_table(self, base_table: np.ndarray, target_size: int) -> np.ndarray:
        """
        Scale quantization table to different block size.
        
        Args:
            base_table: Original 8x8 quantization table
            target_size: Target block size
            
        Returns:
            Scaled quantization table
        """
        if target_size == 8:
            return base_table
        
        # Create frequency-based table for arbitrary size
        table = np.zeros((target_size, target_size), dtype=np.float32)
        
        for i in range(target_size):
            for j in range(target_size):
                # Calculate frequency index (distance from DC component)
                freq_index = i + j
                
                # Map to base table values with interpolation
                if freq_index == 0:
                    table[i, j] = base_table[0, 0]  # DC component
                elif freq_index < 8:
                    # Use diagonal values from base table
                    table[i, j] = base_table[min(i, 7), min(j, 7)]
                else:
                    # Extrapolate for higher frequencies
                    table[i, j] = base_table[7, 7] * (1 + (freq_index - 7) * 0.1)
        
        return table
    
    def get_quantization_table(self) -> np.ndarray:
        """
        Get the current quantization table.
        
        Returns:
            Copy of the quantization table
        """
        return self.quantization_table.copy()
    
    def set_custom_quantization_table(self, table: np.ndarray) -> None:
        """
        Set a custom quantization table.
        
        Args:
            table: Custom quantization table
            
        Raises:
            ValueError: If table shape doesn't match block size or contains invalid values
        """
        if table.shape != (self.block_size, self.block_size):
            raise ValueError(f"Quantization table must be {self.block_size}x{self.block_size}")
        
        if np.any(table <= 0):
            raise ValueError("Quantization table values must be positive")
        
        self.quantization_table = table.astype(np.float32)
    
    def calculate_quantization_error(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Calculate quantization error between original and quantized coefficients.
        
        Args:
            original: Original DCT coefficients
            quantized: Quantized coefficients
            
        Returns:
            Mean squared error between original and dequantized coefficients
        """
        dequantized = self.dequantize(quantized)
        mse = np.mean((original - dequantized) ** 2)
        return float(mse)