"""
Main compression pipeline orchestrator.

This module integrates all components (DCT, quantization, Huffman encoding, preprocessing,
and metrics) into a complete compression/decompression pipeline.
"""

import numpy as np
from typing import Any, Optional, Dict, Tuple
from .models import CompressedData, CompressionMetrics
from .dct_processor import DCTProcessor
from .quantizer import Quantizer
from .huffman_encoder import HuffmanEncoder
from .data_preprocessor import DataPreprocessor
from .metrics_collector import MetricsCollector


class CompressionPipeline:
    """Main orchestrator class integrating all compression components."""
    
    def __init__(self, block_size: int = 8, quality: float = 0.8, 
                 normalization: str = 'minmax'):
        """
        Initialize compression pipeline with configurable parameters.
        
        Args:
            block_size: Size of square blocks for DCT processing (default: 8)
            quality: Quality factor between 0.1 and 1.0 (default: 0.8)
            normalization: Normalization method ('minmax', 'zscore', 'none')
            
        Raises:
            ValueError: If parameters are invalid
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.1 <= quality <= 1.0:
            raise ValueError("quality must be between 0.1 and 1.0")
        if normalization not in ['minmax', 'zscore', 'none']:
            raise ValueError("normalization must be 'minmax', 'zscore', or 'none'")
        
        self.block_size = block_size
        self.quality = quality
        self.normalization = normalization
        
        # Initialize components
        self.dct_processor = DCTProcessor(block_size=block_size)
        self.quantizer = Quantizer(quality=quality, block_size=block_size)
        self.huffman_encoder = HuffmanEncoder()
        self.data_preprocessor = DataPreprocessor()
        self.metrics_collector = MetricsCollector()
        
        # Store last compression metrics
        self.last_metrics: Optional[CompressionMetrics] = None
    
    def compress(self, data: Any) -> CompressedData:
        """
        Main compression method implementing the complete pipeline.
        
        Args:
            data: Input data of various supported types
            
        Returns:
            CompressedData object containing encoded data and metadata
            
        Raises:
            ValueError: If data is invalid or compression fails
        """
        # Start timing
        self.metrics_collector.start_compression_timer()
        
        try:
            # Step 1: Data preprocessing and validation
            validated_data = self.data_preprocessor.validate_and_convert(data)
            original_data = validated_data.copy()
            
            # Step 2: Normalize data
            normalized_data = self.data_preprocessor.normalize_data(
                validated_data, method=self.normalization
            )
            
            # Step 3: Pad data for block processing
            padded_data, original_shape = self.data_preprocessor.pad_to_block_size(
                normalized_data, self.block_size
            )
            
            # Step 4: Apply DCT transformation
            dct_coefficients = self.dct_processor.process_blocks(padded_data, inverse=False)
            
            # Step 5: Quantize DCT coefficients
            quantized_coeffs = self._quantize_blocks(dct_coefficients)
            
            # Step 6: Huffman encode quantized coefficients
            encoded_data, huffman_table = self.huffman_encoder.encode(quantized_coeffs)
            
            # End timing
            compression_time = self.metrics_collector.end_compression_timer()
            
            # Create compressed data object
            compressed_data = CompressedData(
                encoded_data=encoded_data,
                huffman_table=huffman_table,
                original_shape=original_shape,
                block_size=self.block_size,
                quality=self.quality,
                quantization_table=self.quantizer.get_quantization_table(),
                metadata={
                    'normalization_method': self.normalization,
                    'normalization_params': self.data_preprocessor.normalization_params.copy(),
                    'padded_shape': padded_data.shape,
                    'original_dtype': str(validated_data.dtype),
                    'compression_time': compression_time
                }
            )
            
            return compressed_data
            
        except Exception as e:
            # Ensure timer is stopped even if compression fails
            try:
                self.metrics_collector.end_compression_timer()
            except RuntimeError:
                pass  # Timer wasn't started or already stopped
            raise ValueError(f"Compression failed: {str(e)}")
    
    def decompress(self, compressed_data: CompressedData) -> np.ndarray:
        """
        Main decompression method implementing the reverse pipeline.
        
        Args:
            compressed_data: CompressedData object from compression
            
        Returns:
            Reconstructed data as numpy array
            
        Raises:
            ValueError: If compressed data is invalid or decompression fails
        """
        # Start timing
        self.metrics_collector.start_decompression_timer()
        
        try:
            # Validate compressed data
            self._validate_compressed_data(compressed_data)
            
            # Extract metadata
            metadata = compressed_data.metadata
            padded_shape = metadata['padded_shape']
            normalization_method = metadata['normalization_method']
            normalization_params = metadata['normalization_params']
            
            # Step 1: Huffman decode
            quantized_coeffs = self.huffman_encoder.decode(
                compressed_data.encoded_data,
                compressed_data.huffman_table,
                padded_shape
            )
            
            # Step 2: Dequantize coefficients
            # Temporarily set quantization table and parameters
            original_table = self.quantizer.get_quantization_table()
            self.quantizer.set_custom_quantization_table(compressed_data.quantization_table)
            
            dct_coefficients = self._dequantize_blocks(quantized_coeffs)
            
            # Restore original quantization table
            self.quantizer.set_custom_quantization_table(original_table)
            
            # Step 3: Apply inverse DCT
            reconstructed_padded = self.dct_processor.process_blocks(
                dct_coefficients, inverse=True
            )
            
            # Step 4: Remove padding
            reconstructed_normalized = self.data_preprocessor.remove_padding(
                reconstructed_padded, compressed_data.original_shape
            )
            
            # Step 5: Denormalize data
            # Temporarily set normalization parameters
            original_params = self.data_preprocessor.normalization_params.copy()
            self.data_preprocessor.normalization_params = normalization_params
            
            reconstructed_data = self.data_preprocessor.denormalize_data(
                reconstructed_normalized
            )
            
            # Restore original normalization parameters
            self.data_preprocessor.normalization_params = original_params
            
            # End timing
            decompression_time = self.metrics_collector.end_decompression_timer()
            
            # Store decompression time in metadata for metrics calculation
            compressed_data.metadata['decompression_time'] = decompression_time
            
            return reconstructed_data
            
        except Exception as e:
            # Ensure timer is stopped even if decompression fails
            try:
                self.metrics_collector.end_decompression_timer()
            except RuntimeError:
                pass  # Timer wasn't started or already stopped
            raise ValueError(f"Decompression failed: {str(e)}")
    
    def compress_and_measure(self, data: Any) -> Tuple[CompressedData, CompressionMetrics]:
        """
        Compress data and calculate comprehensive metrics.
        
        Args:
            data: Input data to compress
            
        Returns:
            Tuple of (compressed_data, metrics)
        """
        # Store original data for metrics calculation
        original_array = self.data_preprocessor.validate_and_convert(data)
        
        # Compress data
        compressed_data = self.compress(data)
        
        # Decompress for metrics calculation
        reconstructed_data = self.decompress(compressed_data)
        
        # Calculate metrics
        compression_time = compressed_data.metadata.get('compression_time', 0)
        decompression_time = compressed_data.metadata.get('decompression_time', 0)
        
        metrics = self.metrics_collector.create_metrics(
            original_data=original_array,
            reconstructed_data=reconstructed_data,
            compressed_size=len(compressed_data.encoded_data),
            compression_time=compression_time,
            decompression_time=decompression_time
        )
        
        # Store metrics
        self.last_metrics = metrics
        
        return compressed_data, metrics
    
    def _quantize_blocks(self, dct_coefficients: np.ndarray) -> np.ndarray:
        """
        Apply quantization to DCT coefficient blocks.
        
        Args:
            dct_coefficients: DCT coefficients array
            
        Returns:
            Quantized coefficients array
        """
        quantized = np.zeros_like(dct_coefficients, dtype=np.int16)
        
        # Process each block
        for i in range(0, dct_coefficients.shape[0], self.block_size):
            for j in range(0, dct_coefficients.shape[1], self.block_size):
                block = dct_coefficients[i:i+self.block_size, j:j+self.block_size]
                quantized_block = self.quantizer.quantize(block)
                quantized[i:i+self.block_size, j:j+self.block_size] = quantized_block
        
        return quantized
    
    def _dequantize_blocks(self, quantized_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply dequantization to quantized coefficient blocks.
        
        Args:
            quantized_coeffs: Quantized coefficients array
            
        Returns:
            Dequantized coefficients array
        """
        dequantized = np.zeros_like(quantized_coeffs, dtype=np.float32)
        
        # Process each block
        for i in range(0, quantized_coeffs.shape[0], self.block_size):
            for j in range(0, quantized_coeffs.shape[1], self.block_size):
                block = quantized_coeffs[i:i+self.block_size, j:j+self.block_size]
                dequantized_block = self.quantizer.dequantize(block)
                dequantized[i:i+self.block_size, j:j+self.block_size] = dequantized_block
        
        return dequantized
    
    def _validate_compressed_data(self, compressed_data: CompressedData) -> None:
        """
        Validate compressed data structure.
        
        Args:
            compressed_data: CompressedData to validate
            
        Raises:
            ValueError: If compressed data is invalid
        """
        if not isinstance(compressed_data, CompressedData):
            raise ValueError("compressed_data must be CompressedData instance")
        
        if compressed_data.block_size != self.block_size:
            raise ValueError(f"Block size mismatch: expected {self.block_size}, "
                           f"got {compressed_data.block_size}")
        
        required_metadata = ['normalization_method', 'normalization_params', 
                           'padded_shape', 'original_dtype']
        
        for key in required_metadata:
            if key not in compressed_data.metadata:
                raise ValueError(f"Missing required metadata: {key}")
    
    def get_metrics(self) -> Optional[CompressionMetrics]:
        """
        Return performance and quality metrics from last compression.
        
        Returns:
            CompressionMetrics object or None if no compression performed
        """
        return self.last_metrics
    
    def set_quality(self, quality: float) -> None:
        """
        Update quality parameter and regenerate quantization table.
        
        Args:
            quality: New quality factor between 0.1 and 1.0
            
        Raises:
            ValueError: If quality is invalid
        """
        if not 0.1 <= quality <= 1.0:
            raise ValueError("quality must be between 0.1 and 1.0")
        
        self.quality = quality
        self.quantizer = Quantizer(quality=quality, block_size=self.block_size)
    
    def set_block_size(self, block_size: int) -> None:
        """
        Update block size and reinitialize components.
        
        Args:
            block_size: New block size (must be positive)
            
        Raises:
            ValueError: If block_size is invalid
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        
        self.block_size = block_size
        self.dct_processor = DCTProcessor(block_size=block_size)
        self.quantizer = Quantizer(quality=self.quality, block_size=block_size)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about current pipeline configuration.
        
        Returns:
            Dictionary with pipeline configuration
        """
        return {
            'block_size': self.block_size,
            'quality': self.quality,
            'normalization': self.normalization,
            'quantization_table_shape': self.quantizer.get_quantization_table().shape,
            'components': {
                'dct_processor': type(self.dct_processor).__name__,
                'quantizer': type(self.quantizer).__name__,
                'huffman_encoder': type(self.huffman_encoder).__name__,
                'data_preprocessor': type(self.data_preprocessor).__name__,
                'metrics_collector': type(self.metrics_collector).__name__
            }
        }
    
    def batch_compress(self, data_list: list) -> Tuple[list, Dict[str, Any]]:
        """
        Compress multiple data items and collect batch statistics.
        
        Args:
            data_list: List of data items to compress
            
        Returns:
            Tuple of (compressed_data_list, batch_statistics)
        """
        compressed_list = []
        self.metrics_collector.clear_batch_metrics()
        
        for data in data_list:
            compressed_data, metrics = self.compress_and_measure(data)
            compressed_list.append(compressed_data)
            self.metrics_collector.add_batch_metrics(metrics)
        
        batch_stats = self.metrics_collector.get_batch_statistics()
        
        return compressed_list, batch_stats