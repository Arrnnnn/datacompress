"""
DCT (Discrete Cosine Transform) processor for the compression pipeline.

This module implements forward and inverse DCT operations with block-based processing
and padding handling for efficient data transformation.
"""

import numpy as np
from scipy.fft import dct, idct
from typing import Tuple


class DCTProcessor:
    """Handles DCT and inverse DCT operations with block-based processing."""
    
    def __init__(self, block_size: int = 8):
        """
        Initialize DCT processor with specified block size.
        
        Args:
            block_size: Size of square blocks for DCT processing (default: 8)
            
        Raises:
            ValueError: If block_size is not positive or not a power of 2
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if block_size & (block_size - 1) != 0:
            raise ValueError("block_size should be a power of 2 for optimal performance")
            
        self.block_size = block_size
    
    def forward_dct(self, data_block: np.ndarray) -> np.ndarray:
        """
        Apply 2D DCT to a data block.
        
        Args:
            data_block: 2D numpy array of shape (block_size, block_size)
            
        Returns:
            DCT coefficients as 2D numpy array
            
        Raises:
            ValueError: If data_block shape doesn't match block_size
        """
        if data_block.shape != (self.block_size, self.block_size):
            raise ValueError(f"data_block must be {self.block_size}x{self.block_size}")
        
        # Apply 2D DCT using scipy's DCT type-II (standard DCT)
        # Apply DCT along both axes
        dct_coeffs = dct(dct(data_block, axis=0, norm='ortho'), axis=1, norm='ortho')
        return dct_coeffs.astype(np.float32)
    
    def inverse_dct(self, dct_block: np.ndarray) -> np.ndarray:
        """
        Apply inverse 2D DCT to reconstruct data block.
        
        Args:
            dct_block: 2D numpy array of DCT coefficients
            
        Returns:
            Reconstructed data block as 2D numpy array
            
        Raises:
            ValueError: If dct_block shape doesn't match block_size
        """
        if dct_block.shape != (self.block_size, self.block_size):
            raise ValueError(f"dct_block must be {self.block_size}x{self.block_size}")
        
        # Apply inverse 2D DCT using scipy's IDCT type-II
        # Apply IDCT along both axes
        reconstructed = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
        return reconstructed.astype(np.float32)
    
    def process_blocks(self, data: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Process data in blocks with automatic padding handling.
        
        Args:
            data: Input 2D numpy array
            inverse: If True, apply inverse DCT; if False, apply forward DCT
            
        Returns:
            Processed data with same shape as input (padding removed if added)
            
        Raises:
            ValueError: If data is not 2D
        """
        if data.ndim != 2:
            raise ValueError("data must be 2D array")
        
        original_shape = data.shape
        
        # Calculate padding needed to make dimensions divisible by block_size
        pad_height = (self.block_size - (data.shape[0] % self.block_size)) % self.block_size
        pad_width = (self.block_size - (data.shape[1] % self.block_size)) % self.block_size
        
        # Apply padding if needed
        if pad_height > 0 or pad_width > 0:
            padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='edge')
        else:
            padded_data = data.copy()
        
        # Process blocks
        result = np.zeros_like(padded_data, dtype=np.float32)
        
        for i in range(0, padded_data.shape[0], self.block_size):
            for j in range(0, padded_data.shape[1], self.block_size):
                block = padded_data[i:i+self.block_size, j:j+self.block_size]
                
                if inverse:
                    processed_block = self.inverse_dct(block)
                else:
                    processed_block = self.forward_dct(block)
                
                result[i:i+self.block_size, j:j+self.block_size] = processed_block
        
        # Remove padding to return to original shape
        return result[:original_shape[0], :original_shape[1]]
    
    def _validate_block_dimensions(self, data: np.ndarray) -> Tuple[int, int]:
        """
        Validate and return the number of blocks in each dimension.
        
        Args:
            data: Input 2D array
            
        Returns:
            Tuple of (num_blocks_height, num_blocks_width)
        """
        height, width = data.shape
        
        # Calculate number of blocks (with padding consideration)
        num_blocks_h = (height + self.block_size - 1) // self.block_size
        num_blocks_w = (width + self.block_size - 1) // self.block_size
        
        return num_blocks_h, num_blocks_w