"""
Data models for the compression pipeline.

This module defines the core data structures used throughout the compression pipeline,
including compressed data containers, metrics tracking, and Huffman tree nodes.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class CompressedData:
    """Container for compressed data and associated metadata."""
    
    encoded_data: bytes
    huffman_table: Dict[int, str]
    original_shape: Tuple[int, ...]
    block_size: int
    quality: float
    quantization_table: np.ndarray
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate compressed data structure after initialization."""
        if not isinstance(self.encoded_data, bytes):
            raise TypeError("encoded_data must be bytes")
        if not isinstance(self.huffman_table, dict):
            raise TypeError("huffman_table must be a dictionary")
        if not isinstance(self.original_shape, tuple):
            raise TypeError("original_shape must be a tuple")
        if not isinstance(self.quantization_table, np.ndarray):
            raise TypeError("quantization_table must be a numpy array")


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression performance and quality."""
    
    compression_ratio: float
    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float
    mse: float  # Mean Squared Error
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if self.original_size <= 0 or self.compressed_size <= 0:
            raise ValueError("sizes must be positive")
        if self.compression_time < 0 or self.decompression_time < 0:
            raise ValueError("times must be non-negative")


@dataclass
class HuffmanNode:
    """Node structure for Huffman tree construction."""
    
    value: Optional[int]
    frequency: int
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.left is None and self.right is None
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        """Enable comparison for priority queue operations."""
        return self.frequency < other.frequency
    
    def __post_init__(self):
        """Validate node structure after initialization."""
        if self.frequency < 0:
            raise ValueError("frequency must be non-negative")
        if self.is_leaf() and self.value is None:
            raise ValueError("leaf nodes must have a value")
        if not self.is_leaf() and self.value is not None:
            raise ValueError("internal nodes should not have a value")