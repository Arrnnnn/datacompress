"""
Data preprocessing utilities for the compression pipeline.

This module handles input validation, type conversion, normalization, and padding
for different data types including integers, floats, text, and binary data.
"""

import numpy as np
from typing import Union, Tuple, Any, Dict
import struct


class DataPreprocessor:
    """Handles data preprocessing for compression pipeline."""
    
    def __init__(self):
        """Initialize data preprocessor."""
        self.normalization_params: Dict[str, Any] = {}
    
    def validate_and_convert(self, data: Any) -> np.ndarray:
        """
        Validate input data and convert to appropriate numpy array format.
        
        Args:
            data: Input data of various types
            
        Returns:
            Validated and converted numpy array
            
        Raises:
            ValueError: If data type is unsupported or invalid
            TypeError: If data cannot be converted to numpy array
        """
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # Handle different input types
        if isinstance(data, np.ndarray):
            return self._validate_numpy_array(data)
        elif isinstance(data, (list, tuple)):
            return self._convert_sequence(data)
        elif isinstance(data, str):
            return self._convert_text(data)
        elif isinstance(data, bytes):
            return self._convert_binary(data)
        elif isinstance(data, (int, float)):
            return self._convert_scalar(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _validate_numpy_array(self, data: np.ndarray) -> np.ndarray:
        """
        Validate numpy array input.
        
        Args:
            data: Input numpy array
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If array properties are invalid
        """
        if data.size == 0:
            raise ValueError("Input array cannot be empty")
        
        if data.ndim == 0:
            # Convert scalar to 1D array
            data = np.array([data])
        elif data.ndim > 3:
            raise ValueError("Arrays with more than 3 dimensions are not supported")
        
        # Ensure we have a numeric type
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"Array must contain numeric data, got {data.dtype}")
        
        return data.copy()
    
    def _convert_sequence(self, data: Union[list, tuple]) -> np.ndarray:
        """
        Convert list or tuple to numpy array.
        
        Args:
            data: Input sequence
            
        Returns:
            Converted numpy array
        """
        if len(data) == 0:
            raise ValueError("Input sequence cannot be empty")
        
        try:
            array = np.array(data)
            return self._validate_numpy_array(array)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert sequence to numpy array: {e}")
    
    def _convert_text(self, data: str) -> np.ndarray:
        """
        Convert text string to numeric representation.
        
        Args:
            data: Input text string
            
        Returns:
            Numpy array of character codes
        """
        if len(data) == 0:
            raise ValueError("Input text cannot be empty")
        
        # Convert to UTF-8 bytes then to integers
        try:
            byte_data = data.encode('utf-8')
            return np.frombuffer(byte_data, dtype=np.uint8)
        except UnicodeEncodeError as e:
            raise ValueError(f"Cannot encode text to UTF-8: {e}")
    
    def _convert_binary(self, data: bytes) -> np.ndarray:
        """
        Convert binary data to numpy array.
        
        Args:
            data: Input binary data
            
        Returns:
            Numpy array of byte values
        """
        if len(data) == 0:
            raise ValueError("Input binary data cannot be empty")
        
        return np.frombuffer(data, dtype=np.uint8)
    
    def _convert_scalar(self, data: Union[int, float]) -> np.ndarray:
        """
        Convert scalar value to numpy array.
        
        Args:
            data: Input scalar value
            
        Returns:
            1D numpy array containing the scalar
        """
        return np.array([data])
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data to appropriate range for compression.
        
        Args:
            data: Input numpy array
            method: Normalization method ('minmax', 'zscore', 'none')
            
        Returns:
            Normalized data array
            
        Raises:
            ValueError: If normalization method is unsupported
        """
        if method == 'none':
            self.normalization_params = {'method': 'none'}
            return data.copy()
        
        if method == 'minmax':
            return self._minmax_normalize(data)
        elif method == 'zscore':
            return self._zscore_normalize(data)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to scale data to [0, 255] range.
        
        Args:
            data: Input data array
            
        Returns:
            Min-max normalized data
        """
        data_min = np.min(data)
        data_max = np.max(data)
        
        # Store parameters for denormalization
        self.normalization_params = {
            'method': 'minmax',
            'min': float(data_min),
            'max': float(data_max),
            'original_dtype': data.dtype
        }
        
        # Handle constant data
        if data_max == data_min:
            return np.full_like(data, 128, dtype=np.uint8)
        
        # Scale to [0, 255] range
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        return normalized
    
    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization and scale to [0, 255] range.
        
        Args:
            data: Input data array
            
        Returns:
            Z-score normalized and scaled data
        """
        mean = np.mean(data)
        std = np.std(data)
        
        # Store parameters for denormalization
        self.normalization_params = {
            'method': 'zscore',
            'mean': float(mean),
            'std': float(std),
            'original_dtype': data.dtype
        }
        
        # Handle constant data
        if std == 0:
            return np.full_like(data, 128, dtype=np.uint8)
        
        # Z-score normalization
        zscore = (data - mean) / std
        
        # Clip to reasonable range and scale to [0, 255]
        clipped = np.clip(zscore, -3, 3)  # 3 standard deviations
        normalized = ((clipped + 3) / 6 * 255).astype(np.uint8)
        
        return normalized
    
    def denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization to restore original data range.
        
        Args:
            normalized_data: Normalized data array
            
        Returns:
            Denormalized data array
            
        Raises:
            ValueError: If normalization parameters are missing
        """
        if not self.normalization_params:
            raise ValueError("No normalization parameters available for denormalization")
        
        method = self.normalization_params['method']
        
        if method == 'none':
            return normalized_data.copy()
        elif method == 'minmax':
            return self._minmax_denormalize(normalized_data)
        elif method == 'zscore':
            return self._zscore_denormalize(normalized_data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _minmax_denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Reverse min-max normalization.
        
        Args:
            normalized_data: Min-max normalized data
            
        Returns:
            Denormalized data
        """
        params = self.normalization_params
        data_min = params['min']
        data_max = params['max']
        original_dtype = params['original_dtype']
        
        # Handle constant data case
        if data_max == data_min:
            return np.full_like(normalized_data, data_min, dtype=original_dtype)
        
        # Reverse scaling
        denormalized = (normalized_data.astype(np.float64) / 255.0 * 
                       (data_max - data_min) + data_min)
        
        return denormalized.astype(original_dtype)
    
    def _zscore_denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Reverse z-score normalization.
        
        Args:
            normalized_data: Z-score normalized data
            
        Returns:
            Denormalized data
        """
        params = self.normalization_params
        mean = params['mean']
        std = params['std']
        original_dtype = params['original_dtype']
        
        # Handle constant data case
        if std == 0:
            return np.full_like(normalized_data, mean, dtype=original_dtype)
        
        # Reverse scaling and z-score
        scaled = normalized_data.astype(np.float64) / 255.0 * 6 - 3
        denormalized = scaled * std + mean
        
        return denormalized.astype(original_dtype)
    
    def pad_to_block_size(self, data: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """
        Pad data to ensure dimensions are divisible by block size.
        
        Args:
            data: Input data array
            block_size: Required block size
            
        Returns:
            Tuple of (padded_data, original_shape)
        """
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        
        original_shape = data.shape
        
        if data.ndim == 1:
            # For 1D data, pad to make length divisible by block_size
            pad_length = (block_size - (len(data) % block_size)) % block_size
            if pad_length > 0:
                padded = np.pad(data, (0, pad_length), mode='edge')
            else:
                padded = data.copy()
        
        elif data.ndim == 2:
            # For 2D data, pad both dimensions
            pad_height = (block_size - (data.shape[0] % block_size)) % block_size
            pad_width = (block_size - (data.shape[1] % block_size)) % block_size
            
            if pad_height > 0 or pad_width > 0:
                padded = np.pad(data, ((0, pad_height), (0, pad_width)), mode='edge')
            else:
                padded = data.copy()
        
        else:
            raise ValueError("Padding for >2D arrays not implemented")
        
        return padded, original_shape
    
    def remove_padding(self, padded_data: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Remove padding to restore original data shape.
        
        Args:
            padded_data: Padded data array
            original_shape: Original shape before padding
            
        Returns:
            Data with padding removed
        """
        if len(original_shape) == 1:
            return padded_data[:original_shape[0]]
        elif len(original_shape) == 2:
            return padded_data[:original_shape[0], :original_shape[1]]
        else:
            raise ValueError("Unpadding for >2D arrays not implemented")
    
    def get_data_info(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get information about data properties.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with data information
        """
        return {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size,
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'memory_usage': data.nbytes
        }
    
    def convert_back_to_original_type(self, data: np.ndarray, target_type: str) -> Any:
        """
        Convert processed data back to original type format.
        
        Args:
            data: Processed numpy array
            target_type: Target type ('text', 'binary', 'array', 'scalar')
            
        Returns:
            Data converted to target type
            
        Raises:
            ValueError: If target type is unsupported
        """
        if target_type == 'text':
            # Convert back to text string
            byte_data = data.astype(np.uint8).tobytes()
            try:
                return byte_data.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                return byte_data.decode('latin-1', errors='replace')
        
        elif target_type == 'binary':
            # Convert back to bytes
            return data.astype(np.uint8).tobytes()
        
        elif target_type == 'array':
            # Return as numpy array
            return data
        
        elif target_type == 'scalar':
            # Return single value if data has only one element
            if data.size == 1:
                return data.item()
            else:
                raise ValueError("Cannot convert multi-element array to scalar")
        
        else:
            raise ValueError(f"Unsupported target type: {target_type}")