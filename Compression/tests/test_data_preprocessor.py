"""
Unit tests for data preprocessing utilities.
"""

import unittest
import numpy as np
from compression_pipeline.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        self.assertEqual(preprocessor.normalization_params, {})
    
    def test_validate_numpy_array(self):
        """Test numpy array validation."""
        # Valid 2D array
        data = np.array([[1, 2], [3, 4]])
        result = self.preprocessor.validate_and_convert(data)
        np.testing.assert_array_equal(result, data)
        
        # Valid 1D array
        data_1d = np.array([1, 2, 3])
        result = self.preprocessor.validate_and_convert(data_1d)
        np.testing.assert_array_equal(result, data_1d)
        
        # Scalar converted to 1D
        scalar = np.array(5)
        result = self.preprocessor.validate_and_convert(scalar)
        np.testing.assert_array_equal(result, np.array([5]))
    
    def test_invalid_numpy_arrays(self):
        """Test error handling for invalid numpy arrays."""
        # Empty array
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert(np.array([]))
        
        # Too many dimensions
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert(np.random.rand(2, 2, 2, 2))
        
        # Non-numeric data
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert(np.array(['a', 'b']))
    
    def test_convert_list_tuple(self):
        """Test conversion of lists and tuples."""
        # List conversion
        data_list = [1, 2, 3, 4]
        result = self.preprocessor.validate_and_convert(data_list)
        np.testing.assert_array_equal(result, np.array(data_list))
        
        # Tuple conversion
        data_tuple = (1.5, 2.5, 3.5)
        result = self.preprocessor.validate_and_convert(data_tuple)
        np.testing.assert_array_equal(result, np.array(data_tuple))
        
        # Nested list (2D)
        nested_list = [[1, 2], [3, 4]]
        result = self.preprocessor.validate_and_convert(nested_list)
        np.testing.assert_array_equal(result, np.array(nested_list))
    
    def test_invalid_sequences(self):
        """Test error handling for invalid sequences."""
        # Empty list
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert([])
        
        # Mixed types that can't be converted
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert([1, 'a', 3])
    
    def test_convert_text(self):
        """Test text string conversion."""
        text = "Hello, World!"
        result = self.preprocessor.validate_and_convert(text)
        
        # Should convert to byte array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.uint8)
        
        # Should be able to convert back
        converted_back = self.preprocessor.convert_back_to_original_type(result, 'text')
        self.assertEqual(converted_back, text)
    
    def test_convert_text_unicode(self):
        """Test text conversion with Unicode characters."""
        text = "Hello, 世界! 🌍"
        result = self.preprocessor.validate_and_convert(text)
        
        # Should handle Unicode properly
        self.assertIsInstance(result, np.ndarray)
        converted_back = self.preprocessor.convert_back_to_original_type(result, 'text')
        self.assertEqual(converted_back, text)
    
    def test_convert_binary(self):
        """Test binary data conversion."""
        binary_data = b"\\x00\\x01\\x02\\x03\\xff"
        result = self.preprocessor.validate_and_convert(binary_data)
        
        # Should convert to uint8 array
        self.assertEqual(result.dtype, np.uint8)
        
        # Should be able to convert back
        converted_back = self.preprocessor.convert_back_to_original_type(result, 'binary')
        self.assertEqual(converted_back, binary_data)
    
    def test_convert_scalar(self):
        """Test scalar conversion."""
        # Integer scalar
        int_scalar = 42
        result = self.preprocessor.validate_and_convert(int_scalar)
        np.testing.assert_array_equal(result, np.array([42]))
        
        # Float scalar
        float_scalar = 3.14
        result = self.preprocessor.validate_and_convert(float_scalar)
        np.testing.assert_array_equal(result, np.array([3.14]))
    
    def test_invalid_data_types(self):
        """Test error handling for invalid data types."""
        # None input
        with self.assertRaises(ValueError):
            self.preprocessor.validate_and_convert(None)
        
        # Unsupported type
        with self.assertRaises(TypeError):
            self.preprocessor.validate_and_convert({'key': 'value'})
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        normalized = self.preprocessor.normalize_data(data, method='minmax')
        
        # Should be in [0, 255] range
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(np.min(normalized), 0)
        self.assertEqual(np.max(normalized), 255)
        
        # Should be able to denormalize
        denormalized = self.preprocessor.denormalize_data(normalized)
        np.testing.assert_allclose(denormalized, data, rtol=1e-2)
    
    def test_minmax_normalization_constant_data(self):
        """Test min-max normalization with constant data."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = self.preprocessor.normalize_data(data, method='minmax')
        
        # Should handle constant data gracefully
        self.assertTrue(np.all(normalized == 128))
        
        denormalized = self.preprocessor.denormalize_data(normalized)
        np.testing.assert_allclose(denormalized, data)
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        # Create data with known mean and std
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = self.preprocessor.normalize_data(data, method='zscore')
        
        # Should be in [0, 255] range
        self.assertEqual(normalized.dtype, np.uint8)
        
        # Should be able to denormalize
        denormalized = self.preprocessor.denormalize_data(normalized)
        np.testing.assert_allclose(denormalized, data, rtol=1e-1)
    
    def test_zscore_normalization_constant_data(self):
        """Test z-score normalization with constant data."""
        data = np.array([7.0, 7.0, 7.0])
        normalized = self.preprocessor.normalize_data(data, method='zscore')
        
        # Should handle constant data gracefully
        self.assertTrue(np.all(normalized == 128))
        
        denormalized = self.preprocessor.denormalize_data(normalized)
        np.testing.assert_allclose(denormalized, data)
    
    def test_no_normalization(self):
        """Test no normalization option."""
        data = np.array([1, 2, 3, 4, 5])
        normalized = self.preprocessor.normalize_data(data, method='none')
        
        # Should return copy of original data
        np.testing.assert_array_equal(normalized, data)
        
        denormalized = self.preprocessor.denormalize_data(normalized)
        np.testing.assert_array_equal(denormalized, data)
    
    def test_invalid_normalization_method(self):
        """Test error handling for invalid normalization method."""
        data = np.array([1, 2, 3])
        
        with self.assertRaises(ValueError):
            self.preprocessor.normalize_data(data, method='invalid')
    
    def test_denormalization_without_params(self):
        """Test error handling for denormalization without parameters."""
        data = np.array([1, 2, 3])
        
        with self.assertRaises(ValueError):
            self.preprocessor.denormalize_data(data)
    
    def test_padding_1d_data(self):
        """Test padding for 1D data."""
        data = np.array([1, 2, 3, 4, 5])
        block_size = 3
        
        padded, original_shape = self.preprocessor.pad_to_block_size(data, block_size)
        
        # Should pad to make length divisible by block_size
        self.assertEqual(len(padded) % block_size, 0)
        self.assertEqual(original_shape, data.shape)
        
        # Should be able to remove padding
        unpadded = self.preprocessor.remove_padding(padded, original_shape)
        np.testing.assert_array_equal(unpadded, data)
    
    def test_padding_2d_data(self):
        """Test padding for 2D data."""
        data = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array
        block_size = 4
        
        padded, original_shape = self.preprocessor.pad_to_block_size(data, block_size)
        
        # Should pad both dimensions
        self.assertEqual(padded.shape[0] % block_size, 0)
        self.assertEqual(padded.shape[1] % block_size, 0)
        self.assertEqual(original_shape, data.shape)
        
        # Should be able to remove padding
        unpadded = self.preprocessor.remove_padding(padded, original_shape)
        np.testing.assert_array_equal(unpadded, data)
    
    def test_padding_no_padding_needed(self):
        """Test padding when no padding is needed."""
        data = np.array([1, 2, 3, 4])  # Length 4
        block_size = 2
        
        padded, original_shape = self.preprocessor.pad_to_block_size(data, block_size)
        
        # Should not add padding
        np.testing.assert_array_equal(padded, data)
        self.assertEqual(original_shape, data.shape)
    
    def test_invalid_block_size(self):
        """Test error handling for invalid block size."""
        data = np.array([1, 2, 3])
        
        with self.assertRaises(ValueError):
            self.preprocessor.pad_to_block_size(data, 0)
        
        with self.assertRaises(ValueError):
            self.preprocessor.pad_to_block_size(data, -1)
    
    def test_get_data_info(self):
        """Test data information extraction."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        info = self.preprocessor.get_data_info(data)
        
        expected_keys = ['shape', 'dtype', 'size', 'min_value', 'max_value', 
                        'mean', 'std', 'memory_usage']
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['shape'], (2, 2))
        self.assertEqual(info['size'], 4)
        self.assertEqual(info['min_value'], 1.0)
        self.assertEqual(info['max_value'], 4.0)
        self.assertEqual(info['mean'], 2.5)
    
    def test_convert_back_to_original_type(self):
        """Test conversion back to original types."""
        # Test scalar conversion
        data = np.array([42])
        result = self.preprocessor.convert_back_to_original_type(data, 'scalar')
        self.assertEqual(result, 42)
        
        # Test array conversion
        data = np.array([1, 2, 3])
        result = self.preprocessor.convert_back_to_original_type(data, 'array')
        np.testing.assert_array_equal(result, data)
        
        # Test invalid scalar conversion
        multi_element = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            self.preprocessor.convert_back_to_original_type(multi_element, 'scalar')
        
        # Test invalid target type
        with self.assertRaises(ValueError):
            self.preprocessor.convert_back_to_original_type(data, 'invalid')


if __name__ == '__main__':
    unittest.main()