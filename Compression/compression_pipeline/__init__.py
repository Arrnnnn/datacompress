"""
Compression Pipeline - A multi-stage compression system using DCT, quantization, and Huffman encoding.

This package provides efficient data compression for database applications using a three-stage
pipeline inspired by JPEG compression but adapted for general data compression.
"""

"""
Compression Pipeline - A multi-stage compression system using DCT, quantization, and Huffman encoding.

This package provides efficient data compression for database applications using a three-stage
pipeline inspired by JPEG compression but adapted for general data compression.
"""

from .models import CompressedData, CompressionMetrics, HuffmanNode
from .pipeline import CompressionPipeline
from .dct_processor import DCTProcessor
from .quantizer import Quantizer
from .huffman_encoder import HuffmanEncoder
from .data_preprocessor import DataPreprocessor
from .metrics_collector import MetricsCollector
from .performance import PerformanceOptimizer
from .error_handler import ErrorHandler
from . import exceptions

__version__ = "1.0.0"
__author__ = "Compression Pipeline Team"
__email__ = "team@compressionpipeline.com"
__license__ = "MIT"

__all__ = [
    # Core classes
    "CompressionPipeline",
    
    # Data models
    "CompressedData", 
    "CompressionMetrics", 
    "HuffmanNode",
    
    # Component classes
    "DCTProcessor", 
    "Quantizer", 
    "HuffmanEncoder", 
    "DataPreprocessor", 
    "MetricsCollector",
    
    # Utilities
    "PerformanceOptimizer",
    "ErrorHandler",
    
    # Exceptions module
    "exceptions",
]

# Package metadata
__doc__ = """
Compression Pipeline

A comprehensive Python library for data compression using DCT (Discrete Cosine Transform), 
quantization, and Huffman encoding. This pipeline provides efficient lossy compression 
suitable for database storage, scientific data archival, and general-purpose data compression.

Key Features:
- Multi-stage compression pipeline (DCT → Quantization → Huffman)
- Configurable quality settings
- Support for multiple data types
- Comprehensive performance metrics
- Memory-efficient processing
- Database integration capabilities
- Robust error handling

Quick Start:
    >>> import numpy as np
    >>> from compression_pipeline import CompressionPipeline
    >>> 
    >>> # Create sample data
    >>> data = np.random.rand(64, 64).astype(np.float32)
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = CompressionPipeline(quality=0.8)
    >>> 
    >>> # Compress and decompress
    >>> compressed_data = pipeline.compress(data)
    >>> reconstructed_data = pipeline.decompress(compressed_data)
    >>> 
    >>> # Get metrics
    >>> compressed_data, metrics = pipeline.compress_and_measure(data)
    >>> print(f"Compression ratio: {metrics.compression_ratio:.2f}")

For more examples, see the examples/ directory.
"""