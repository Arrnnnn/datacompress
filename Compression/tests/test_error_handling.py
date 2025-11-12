"""
Unit tests for error handling components.
"""

import unittest
import numpy as np
from compression_pipeline.error_handler import ErrorHandler
from compression_pipeline.exceptions import (
    DataValidationError, UnsupportedDataTypeError, InsufficientResourcesError,
    CorruptedDataError, CompressionError, DecompressionError
)
from compression_pipeline.models import CompressedData


class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler(memory_threshold_mb=50.0)
        self.assertEqual(handler.memory_threshold_mb, 50.0)
        self.assertIn('numpy.ndarray', handler.supported_data_types)
    
    def test_validate_valid_numpy_array(self):
        """Test validation of valid numpy arrays."""
        data = np.array([[1, 2], [3, 4]])
        data_info = self.error_handler.validate_input_data(data)
        
        self.assertEqual(data_info['type_name'], 'ndarray')
        self.assertGreater(data_info['size_estimate'], 0)
        self.assertIn('shape', data_info['properties'])
    
    def test_validate_invalid_numpy_arrays(self):
        """Test validation of invalid numpy arrays."""
        # Empty array
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(np.array([]))
        
        # Too many dimensions
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(np.random.rand(2, 2, 2, 2))
        
        # Non-numeric data
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(np.array(['a', 'b']))
        
        # NaN values
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(np.array([1, np.nan, 3]))
        
        # Infinite values
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(np.array([1, np.inf, 3]))
    
    def test_validate_valid_sequences(self):
        """Test validation of valid lists and tuples."""
        # Valid list
        data_info = self.error_handler.validate_input_data([1, 2, 3, 4])
        self.assertEqual(data_info['type_name'], 'list')
        
        # Valid tuple
        data_info = self.error_handler.validate_input_data((1.5, 2.5, 3.5))
        self.assertEqual(data_info['type_name'], 'tuple')
    
    def test_validate_invalid_sequences(self):
        """Test validation of invalid sequences."""
        # Empty list
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data([])
        
        # Mixed types
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data([1, 'a', 3])
    
    def test_validate_valid_text(self):
        """Test validation of valid text strings."""
        data_info = self.error_handler.validate_input_data("Hello, World!")
        self.assertEqual(data_info['type_name'], 'str')
        self.assertGreater(data_info['size_estimate'], 0)
    
    def test_validate_invalid_text(self):
        """Test validation of invalid text."""
        # Empty string
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data("")
    
    def test_validate_valid_binary(self):
        """Test validation of valid binary data."""
        data_info = self.error_handler.validate_input_data(b"\\x00\\x01\\x02")
        self.assertEqual(data_info['type_name'], 'bytes')
        self.assertGreater(data_info['size_estimate'], 0)
    
    def test_validate_invalid_binary(self):
        """Test validation of invalid binary data."""
        # Empty bytes
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(b"")
    
    def test_validate_valid_scalars(self):
        """Test validation of valid scalar values."""
        # Integer
        data_info = self.error_handler.validate_input_data(42)
        self.assertEqual(data_info['type_name'], 'int')
        
        # Float
        data_info = self.error_handler.validate_input_data(3.14)
        self.assertEqual(data_info['type_name'], 'float')
    
    def test_validate_invalid_scalars(self):
        """Test validation of invalid scalar values."""
        # NaN float
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(float('nan'))
        
        # Infinite float
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(float('inf'))
    
    def test_validate_unsupported_types(self):
        """Test validation of unsupported data types."""
        # Dictionary
        with self.assertRaises(UnsupportedDataTypeError) as context:
            self.error_handler.validate_input_data({'key': 'value'})
        
        error = context.exception
        self.assertEqual(error.data_type, 'dict')
        self.assertIn('numpy.ndarray', error.supported_types)
    
    def test_validate_none_data(self):
        """Test validation of None data."""
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_input_data(None)
    
    def test_check_system_resources_sufficient(self):
        """Test resource checking with sufficient resources."""
        # Request small amount of memory
        resource_info = self.error_handler.check_system_resources(1.0)  # 1 MB
        
        self.assertIn('available_memory_mb', resource_info)
        self.assertIn('sufficient_memory', resource_info)
        self.assertTrue(resource_info['sufficient_memory'])
    
    def test_check_system_resources_insufficient(self):
        """Test resource checking with insufficient resources."""
        # Request unreasonably large amount of memory
        with self.assertRaises(InsufficientResourcesError) as context:
            self.error_handler.check_system_resources(1000000.0)  # 1TB
        
        error = context.exception
        self.assertEqual(error.resource_type, 'memory')
        self.assertIsNotNone(error.required)
        self.assertIsNotNone(error.available)
    
    def test_validate_valid_compressed_data(self):
        """Test validation of valid compressed data."""
        # Create valid compressed data
        compressed_data = CompressedData(
            encoded_data=b"test_data",
            huffman_table={1: '0', 2: '1'},
            original_shape=(4, 4),
            block_size=8,
            quality=0.8,
            quantization_table=np.ones((8, 8)),
            metadata={
                'normalization_method': 'minmax',
                'normalization_params': {'method': 'minmax'},
                'padded_shape': (8, 8),
                'original_dtype': 'float32'
            }
        )
        
        # Should not raise any exception
        self.error_handler.validate_compressed_data(compressed_data)
    
    def test_validate_invalid_compressed_data_type(self):
        """Test validation of invalid compressed data type."""
        with self.assertRaises(DataValidationError):
            self.error_handler.validate_compressed_data("not_compressed_data")
    
    def test_validate_corrupted_compressed_data(self):
        """Test validation of corrupted compressed data."""
        # Missing encoded_data
        compressed_data = CompressedData(
            encoded_data=b"",  # Empty data
            huffman_table={1: '0'},
            original_shape=(4, 4),
            block_size=8,
            quality=0.8,
            quantization_table=np.ones((8, 8)),
            metadata={
                'normalization_method': 'minmax',
                'normalization_params': {'method': 'minmax'},
                'padded_shape': (8, 8),
                'original_dtype': 'float32'
            }
        )
        
        with self.assertRaises(CorruptedDataError) as context:
            self.error_handler.validate_compressed_data(compressed_data)
        
        error = context.exception
        self.assertEqual(error.corruption_type, 'empty_data')
    
    def test_validate_compressed_data_missing_metadata(self):
        """Test validation of compressed data with missing metadata."""
        compressed_data = CompressedData(
            encoded_data=b"test_data",
            huffman_table={1: '0'},
            original_shape=(4, 4),
            block_size=8,
            quality=0.8,
            quantization_table=np.ones((8, 8)),
            metadata={}  # Missing required metadata
        )
        
        with self.assertRaises(CorruptedDataError) as context:
            self.error_handler.validate_compressed_data(compressed_data)
        
        error = context.exception
        self.assertEqual(error.corruption_type, 'missing_metadata')
    
    def test_handle_compression_error(self):
        """Test compression error handling."""
        original_error = ValueError("Test error")
        stage = "dct_transform"
        context = {"block_size": 8}
        
        compression_error = self.error_handler.handle_compression_error(
            original_error, stage, context
        )
        
        self.assertIsInstance(compression_error, CompressionError)
        self.assertEqual(compression_error.stage, stage)
        self.assertEqual(compression_error.original_error, original_error)
        self.assertIn(stage, str(compression_error))
        self.assertIn("Test error", str(compression_error))
    
    def test_handle_decompression_error(self):
        """Test decompression error handling."""
        original_error = KeyError("Missing key")
        stage = "huffman_decode"
        context = {"table_size": 10}
        
        decompression_error = self.error_handler.handle_decompression_error(
            original_error, stage, context
        )
        
        self.assertIsInstance(decompression_error, DecompressionError)
        self.assertEqual(decompression_error.stage, stage)
        self.assertEqual(decompression_error.original_error, original_error)
        self.assertIn(stage, str(decompression_error))
    
    def test_get_error_recovery_suggestions_unsupported_type(self):
        """Test recovery suggestions for unsupported data type."""
        error = UnsupportedDataTypeError('dict', ['numpy.ndarray', 'list'])
        suggestions = self.error_handler.get_error_recovery_suggestions(error)
        
        self.assertEqual(suggestions['error_type'], 'UnsupportedDataTypeError')
        self.assertGreater(len(suggestions['suggestions']), 0)
        self.assertIn('numpy.array()', str(suggestions['suggestions']))
    
    def test_get_error_recovery_suggestions_insufficient_resources(self):
        """Test recovery suggestions for insufficient resources."""
        error = InsufficientResourcesError("Not enough memory", "memory")
        suggestions = self.error_handler.get_error_recovery_suggestions(error)
        
        self.assertEqual(suggestions['error_type'], 'InsufficientResourcesError')
        self.assertGreater(len(suggestions['suggestions']), 0)
        self.assertGreater(len(suggestions['alternative_approaches']), 0)
        self.assertIn('smaller chunks', str(suggestions['suggestions']))
    
    def test_get_error_recovery_suggestions_corrupted_data(self):
        """Test recovery suggestions for corrupted data."""
        error = CorruptedDataError("Data is corrupted", "invalid_checksum")
        suggestions = self.error_handler.get_error_recovery_suggestions(error)
        
        self.assertEqual(suggestions['error_type'], 'CorruptedDataError')
        self.assertGreater(len(suggestions['suggestions']), 0)
        self.assertIn('integrity', str(suggestions['suggestions']))
    
    def test_data_analysis(self):
        """Test data analysis functionality."""
        # Test with numpy array
        data = np.random.rand(10, 10)
        data_info = self.error_handler._analyze_data(data)
        
        self.assertEqual(data_info['type_name'], 'ndarray')
        self.assertEqual(data_info['size_estimate'], 100)
        self.assertGreater(data_info['memory_usage_mb'], 0)
        self.assertIn('shape', data_info['properties'])
        
        # Test with list
        data = [1, 2, 3, 4, 5]
        data_info = self.error_handler._analyze_data(data)
        
        self.assertEqual(data_info['type_name'], 'list')
        self.assertEqual(data_info['size_estimate'], 5)
        self.assertIn('length', data_info['properties'])


if __name__ == '__main__':
    unittest.main()