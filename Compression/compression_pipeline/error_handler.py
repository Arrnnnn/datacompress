"""
Error handling utilities for the compression pipeline.

This module provides comprehensive error handling, validation, and recovery
mechanisms for robust operation in production environments.
"""

import sys
import traceback
import psutil
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from .exceptions import (
    CompressionPipelineError, DataValidationError, CompressionError,
    DecompressionError, CorruptedDataError, InsufficientResourcesError,
    ConfigurationError, UnsupportedDataTypeError
)


class ErrorHandler:
    """Comprehensive error handling and validation for the compression pipeline."""
    
    def __init__(self, memory_threshold_mb: float = 100.0):
        """
        Initialize error handler.
        
        Args:
            memory_threshold_mb: Memory threshold in MB for resource checking
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.supported_data_types = [
            'numpy.ndarray', 'list', 'tuple', 'str', 'bytes', 'int', 'float'
        ]
    
    def validate_input_data(self, data: Any) -> Dict[str, Any]:
        """
        Comprehensive input data validation.
        
        Args:
            data: Input data to validate
            
        Returns:
            Dictionary with validation results and data info
            
        Raises:
            DataValidationError: If data validation fails
            UnsupportedDataTypeError: If data type is not supported
        """
        try:
            data_info = self._analyze_data(data)
            
            # Check data type support
            if data_info['type_name'] not in self.supported_data_types:
                raise UnsupportedDataTypeError(
                    data_info['type_name'], 
                    self.supported_data_types
                )
            
            # Check for None or empty data
            if data is None:
                raise DataValidationError("Input data cannot be None")
            
            # Type-specific validation
            if isinstance(data, np.ndarray):
                self._validate_numpy_array(data, data_info)
            elif isinstance(data, (list, tuple)):
                self._validate_sequence(data, data_info)
            elif isinstance(data, str):
                self._validate_text(data, data_info)
            elif isinstance(data, bytes):
                self._validate_binary(data, data_info)
            elif isinstance(data, (int, float)):
                self._validate_scalar(data, data_info)
            
            return data_info
            
        except (DataValidationError, UnsupportedDataTypeError):
            raise
        except Exception as e:
            raise DataValidationError(
                f"Unexpected error during data validation: {str(e)}",
                {'original_error': str(e)}
            )
    
    def _analyze_data(self, data: Any) -> Dict[str, Any]:
        """Analyze data properties for validation."""
        data_info = {
            'type_name': type(data).__name__,
            'size_estimate': 0,
            'memory_usage_mb': 0,
            'properties': {}
        }
        
        try:
            if isinstance(data, np.ndarray):
                data_info.update({
                    'size_estimate': data.size,
                    'memory_usage_mb': data.nbytes / (1024 * 1024),
                    'properties': {
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'ndim': data.ndim
                    }
                })
            elif isinstance(data, (list, tuple)):
                data_info.update({
                    'size_estimate': len(data),
                    'memory_usage_mb': sys.getsizeof(data) / (1024 * 1024),
                    'properties': {
                        'length': len(data),
                        'nested': any(isinstance(item, (list, tuple)) for item in data)
                    }
                })
            elif isinstance(data, str):
                encoded_size = len(data.encode('utf-8'))
                data_info.update({
                    'size_estimate': encoded_size,
                    'memory_usage_mb': encoded_size / (1024 * 1024),
                    'properties': {
                        'length': len(data),
                        'encoded_size': encoded_size
                    }
                })
            elif isinstance(data, bytes):
                data_info.update({
                    'size_estimate': len(data),
                    'memory_usage_mb': len(data) / (1024 * 1024),
                    'properties': {
                        'length': len(data)
                    }
                })
            elif isinstance(data, (int, float)):
                data_info.update({
                    'size_estimate': 1,
                    'memory_usage_mb': sys.getsizeof(data) / (1024 * 1024),
                    'properties': {
                        'value': data
                    }
                })
        except Exception as e:
            data_info['analysis_error'] = str(e)
        
        return data_info
    
    def _validate_numpy_array(self, data: np.ndarray, data_info: Dict[str, Any]) -> None:
        """Validate numpy array specific properties."""
        if data.size == 0:
            raise DataValidationError("Numpy array cannot be empty", data_info)
        
        if data.ndim > 3:
            raise DataValidationError(
                f"Arrays with more than 3 dimensions are not supported (got {data.ndim}D)",
                data_info
            )
        
        if not np.issubdtype(data.dtype, np.number):
            raise DataValidationError(
                f"Array must contain numeric data (got {data.dtype})",
                data_info
            )
        
        # Check for invalid values
        if np.any(np.isnan(data)):
            raise DataValidationError("Array contains NaN values", data_info)
        
        if np.any(np.isinf(data)):
            raise DataValidationError("Array contains infinite values", data_info)
    
    def _validate_sequence(self, data: Union[list, tuple], data_info: Dict[str, Any]) -> None:
        """Validate list/tuple specific properties."""
        if len(data) == 0:
            raise DataValidationError("Sequence cannot be empty", data_info)
        
        # Check if all elements are numeric
        try:
            np.array(data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise DataValidationError(
                f"Sequence contains non-numeric data: {str(e)}",
                data_info
            )
    
    def _validate_text(self, data: str, data_info: Dict[str, Any]) -> None:
        """Validate text string properties."""
        if len(data) == 0:
            raise DataValidationError("Text string cannot be empty", data_info)
        
        # Check if text can be encoded to UTF-8
        try:
            data.encode('utf-8')
        except UnicodeEncodeError as e:
            raise DataValidationError(
                f"Text cannot be encoded to UTF-8: {str(e)}",
                data_info
            )
    
    def _validate_binary(self, data: bytes, data_info: Dict[str, Any]) -> None:
        """Validate binary data properties."""
        if len(data) == 0:
            raise DataValidationError("Binary data cannot be empty", data_info)
    
    def _validate_scalar(self, data: Union[int, float], data_info: Dict[str, Any]) -> None:
        """Validate scalar value properties."""
        if isinstance(data, float):
            if np.isnan(data):
                raise DataValidationError("Scalar value cannot be NaN", data_info)
            if np.isinf(data):
                raise DataValidationError("Scalar value cannot be infinite", data_info)
    
    def check_system_resources(self, estimated_memory_mb: float) -> Dict[str, Any]:
        """
        Check if system has sufficient resources for operation.
        
        Args:
            estimated_memory_mb: Estimated memory requirement in MB
            
        Returns:
            Dictionary with resource information
            
        Raises:
            InsufficientResourcesError: If resources are insufficient
        """
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            resource_info = {
                'available_memory_mb': available_mb,
                'estimated_requirement_mb': estimated_memory_mb,
                'memory_threshold_mb': self.memory_threshold_mb,
                'sufficient_memory': available_mb >= estimated_memory_mb + self.memory_threshold_mb
            }
            
            # Check memory availability
            if not resource_info['sufficient_memory']:
                raise InsufficientResourcesError(
                    f"Insufficient memory: need {estimated_memory_mb:.1f}MB + {self.memory_threshold_mb:.1f}MB buffer, "
                    f"but only {available_mb:.1f}MB available",
                    resource_type="memory",
                    required=f"{estimated_memory_mb + self.memory_threshold_mb:.1f}MB",
                    available=f"{available_mb:.1f}MB"
                )
            
            return resource_info
            
        except psutil.Error as e:
            # If we can't check resources, issue a warning but don't fail
            return {
                'resource_check_error': str(e),
                'warning': 'Could not verify system resources'
            }
    
    def validate_compressed_data(self, compressed_data: Any) -> None:
        """
        Validate compressed data structure for decompression.
        
        Args:
            compressed_data: Compressed data object to validate
            
        Raises:
            CorruptedDataError: If compressed data appears corrupted
            DataValidationError: If data structure is invalid
        """
        from .models import CompressedData
        
        if not isinstance(compressed_data, CompressedData):
            raise DataValidationError(
                f"Expected CompressedData object, got {type(compressed_data).__name__}"
            )
        
        # Check required fields
        required_fields = [
            'encoded_data', 'huffman_table', 'original_shape', 
            'block_size', 'quality', 'quantization_table', 'metadata'
        ]
        
        for field in required_fields:
            if not hasattr(compressed_data, field):
                raise CorruptedDataError(
                    f"Missing required field: {field}",
                    corruption_type="missing_field"
                )
        
        # Validate encoded data
        if not isinstance(compressed_data.encoded_data, bytes):
            raise CorruptedDataError(
                "encoded_data must be bytes",
                corruption_type="invalid_type"
            )
        
        if len(compressed_data.encoded_data) == 0:
            raise CorruptedDataError(
                "encoded_data cannot be empty",
                corruption_type="empty_data"
            )
        
        # Validate Huffman table
        if not isinstance(compressed_data.huffman_table, dict):
            raise CorruptedDataError(
                "huffman_table must be a dictionary",
                corruption_type="invalid_type"
            )
        
        if len(compressed_data.huffman_table) == 0:
            raise CorruptedDataError(
                "huffman_table cannot be empty",
                corruption_type="empty_table"
            )
        
        # Validate metadata
        required_metadata = [
            'normalization_method', 'normalization_params', 
            'padded_shape', 'original_dtype'
        ]
        
        for key in required_metadata:
            if key not in compressed_data.metadata:
                raise CorruptedDataError(
                    f"Missing required metadata: {key}",
                    corruption_type="missing_metadata"
                )
    
    def handle_compression_error(self, error: Exception, stage: str, 
                                context: Dict[str, Any] = None) -> CompressionError:
        """
        Handle and wrap compression errors with additional context.
        
        Args:
            error: Original exception
            stage: Pipeline stage where error occurred
            context: Additional context information
            
        Returns:
            CompressionError with enhanced information
        """
        context = context or {}
        
        # Create detailed error message
        message = f"Compression failed at stage '{stage}': {str(error)}"
        
        if context:
            message += f" (Context: {context})"
        
        return CompressionError(message, stage=stage, original_error=error)
    
    def handle_decompression_error(self, error: Exception, stage: str,
                                  context: Dict[str, Any] = None) -> DecompressionError:
        """
        Handle and wrap decompression errors with additional context.
        
        Args:
            error: Original exception
            stage: Pipeline stage where error occurred
            context: Additional context information
            
        Returns:
            DecompressionError with enhanced information
        """
        context = context or {}
        
        # Create detailed error message
        message = f"Decompression failed at stage '{stage}': {str(error)}"
        
        if context:
            message += f" (Context: {context})"
        
        return DecompressionError(message, stage=stage, original_error=error)
    
    def get_error_recovery_suggestions(self, error: CompressionPipelineError) -> Dict[str, Any]:
        """
        Provide recovery suggestions based on error type.
        
        Args:
            error: Pipeline error
            
        Returns:
            Dictionary with recovery suggestions
        """
        suggestions = {
            'error_type': type(error).__name__,
            'suggestions': [],
            'alternative_approaches': []
        }
        
        if isinstance(error, UnsupportedDataTypeError):
            suggestions['suggestions'].extend([
                f"Convert data to supported type: {', '.join(error.supported_types)}",
                "Use numpy.array() to convert sequences to numpy arrays",
                "Ensure data contains only numeric values"
            ])
        
        elif isinstance(error, InsufficientResourcesError):
            suggestions['suggestions'].extend([
                "Process data in smaller chunks",
                "Increase available system memory",
                "Use a smaller block size to reduce memory usage",
                "Close other applications to free memory"
            ])
            suggestions['alternative_approaches'].extend([
                "Use streaming compression for large datasets",
                "Implement disk-based processing"
            ])
        
        elif isinstance(error, CorruptedDataError):
            suggestions['suggestions'].extend([
                "Verify data integrity before decompression",
                "Check if compression completed successfully",
                "Ensure data wasn't modified during transmission/storage"
            ])
            suggestions['alternative_approaches'].extend([
                "Re-compress the original data",
                "Use error correction codes for data transmission"
            ])
        
        elif isinstance(error, ConfigurationError):
            suggestions['suggestions'].extend([
                "Check parameter values are within valid ranges",
                "Use default configuration for initial testing",
                "Refer to documentation for parameter guidelines"
            ])
        
        return suggestions