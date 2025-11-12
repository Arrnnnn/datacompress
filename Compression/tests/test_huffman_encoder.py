"""
Unit tests for Huffman encoding component.
"""

import unittest
import numpy as np
from compression_pipeline.huffman_encoder import HuffmanEncoder
from compression_pipeline.models import HuffmanNode


class TestHuffmanEncoder(unittest.TestCase):
    """Test cases for HuffmanEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = HuffmanEncoder()
    
    def test_initialization(self):
        """Test HuffmanEncoder initialization."""
        encoder = HuffmanEncoder()
        self.assertIsNone(encoder.code_table)
        self.assertIsNone(encoder.decode_table)
    
    def test_frequency_table_building(self):
        """Test frequency table construction."""
        data = np.array([1, 2, 1, 3, 1, 2])
        frequencies = self.encoder._build_frequency_table(data)
        
        expected = {1: 3, 2: 2, 3: 1}
        self.assertEqual(frequencies, expected)
    
    def test_huffman_tree_construction(self):
        """Test Huffman tree building."""
        frequencies = {1: 3, 2: 2, 3: 1}
        root = self.encoder._build_huffman_tree(frequencies)
        
        # Check root properties
        self.assertIsNotNone(root)
        self.assertEqual(root.frequency, 6)  # Sum of all frequencies
        self.assertFalse(root.is_leaf())
        
        # Tree should have internal nodes and leaves
        self.assertIsNotNone(root.left)
        self.assertIsNotNone(root.right)
    
    def test_huffman_tree_single_symbol(self):
        """Test Huffman tree with single symbol."""
        frequencies = {5: 10}
        root = self.encoder._build_huffman_tree(frequencies)
        
        self.assertTrue(root.is_leaf())
        self.assertEqual(root.value, 5)
        self.assertEqual(root.frequency, 10)
    
    def test_code_generation(self):
        """Test Huffman code generation."""
        frequencies = {1: 3, 2: 2, 3: 1}
        root = self.encoder._build_huffman_tree(frequencies)
        codes = self.encoder._generate_codes(root)
        
        # Check that all symbols have codes
        self.assertIn(1, codes)
        self.assertIn(2, codes)
        self.assertIn(3, codes)
        
        # Check that codes are binary strings
        for code in codes.values():
            self.assertTrue(all(c in '01' for c in code))
        
        # More frequent symbols should have shorter codes
        # (This is a general property but may not always hold due to tree structure)
        self.assertLessEqual(len(codes[1]), len(codes[3]))
    
    def test_code_generation_single_symbol(self):
        """Test code generation for single symbol."""
        root = HuffmanNode(value=5, frequency=10)
        codes = self.encoder._generate_codes(root)
        
        self.assertEqual(codes, {5: '0'})
    
    def test_bits_to_bytes_conversion(self):
        """Test conversion from bit string to bytes."""
        bit_string = '10110010'
        encoded_bytes = self.encoder._bits_to_bytes(bit_string)
        
        # Should have padding info + data bytes
        self.assertGreater(len(encoded_bytes), 0)
        
        # Convert back and check
        decoded_bits = self.encoder._bytes_to_bits(encoded_bytes)
        self.assertEqual(bit_string, decoded_bits)
    
    def test_bits_to_bytes_with_padding(self):
        """Test bit string conversion with padding needed."""
        bit_string = '101100'  # 6 bits, needs 2 bits padding
        encoded_bytes = self.encoder._bits_to_bytes(bit_string)
        decoded_bits = self.encoder._bytes_to_bits(encoded_bytes)
        
        self.assertEqual(bit_string, decoded_bits)
    
    def test_encode_decode_roundtrip(self):
        """Test complete encode/decode cycle."""
        # Create test data
        original_data = np.array([[1, 2, 1], [3, 1, 2]], dtype=np.int32)
        
        # Encode
        encoded_bytes, code_table = self.encoder.encode(original_data)
        
        # Decode
        decoded_data = self.encoder.decode(encoded_bytes, code_table, original_data.shape)
        
        # Check reconstruction
        np.testing.assert_array_equal(original_data, decoded_data)
    
    def test_encode_single_symbol(self):
        """Test encoding data with single unique symbol."""
        data = np.array([5, 5, 5, 5])
        
        encoded_bytes, code_table = self.encoder.encode(data)
        decoded_data = self.encoder.decode(encoded_bytes, code_table, data.shape)
        
        np.testing.assert_array_equal(data, decoded_data)
        self.assertEqual(code_table, {5: '0'})
    
    def test_encode_empty_data(self):
        """Test error handling for empty data."""
        with self.assertRaises(ValueError):
            self.encoder.encode(np.array([]))
    
    def test_decode_empty_code_table(self):
        """Test error handling for empty code table."""
        with self.assertRaises(ValueError):
            self.encoder.decode(b'test', {}, (2, 2))
    
    def test_decode_invalid_encoding(self):
        """Test error handling for invalid encoded data."""
        # Create valid encoding first
        data = np.array([1, 2, 3])
        encoded_bytes, code_table = self.encoder.encode(data)
        
        # Try to decode with wrong expected length
        with self.assertRaises(ValueError):
            self.encoder.decode(encoded_bytes, code_table, (10,))  # Wrong length
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        # Create data with repeated patterns (should compress well)
        data = np.array([1, 1, 1, 1, 2, 2, 3] * 10, dtype=np.int32)
        
        encoded_bytes, _ = self.encoder.encode(data)
        ratio = self.encoder.calculate_compression_ratio(data, encoded_bytes)
        
        # Should achieve some compression
        self.assertGreater(ratio, 1.0)
    
    def test_code_statistics(self):
        """Test code statistics generation."""
        data = np.array([1, 2, 1, 3, 1, 2])
        self.encoder.encode(data)
        
        stats = self.encoder.get_code_statistics()
        
        self.assertIn('num_symbols', stats)
        self.assertIn('min_code_length', stats)
        self.assertIn('max_code_length', stats)
        self.assertIn('avg_code_length', stats)
        
        self.assertEqual(stats['num_symbols'], 3)  # 1, 2, 3
        self.assertGreater(stats['min_code_length'], 0)
        self.assertGreaterEqual(stats['max_code_length'], stats['min_code_length'])
    
    def test_large_data_encoding(self):
        """Test encoding of larger datasets."""
        # Create larger test data
        np.random.seed(42)  # For reproducible results
        data = np.random.randint(0, 10, size=(100, 100))
        
        encoded_bytes, code_table = self.encoder.encode(data)
        decoded_data = self.encoder.decode(encoded_bytes, code_table, data.shape)
        
        np.testing.assert_array_equal(data, decoded_data)
    
    def test_different_data_types(self):
        """Test encoding different integer data types."""
        # Test with different integer types
        data_int8 = np.array([1, 2, 3], dtype=np.int8)
        data_int16 = np.array([1000, 2000, 3000], dtype=np.int16)
        
        # Should work with different integer types
        encoded_bytes, code_table = self.encoder.encode(data_int8)
        decoded_data = self.encoder.decode(encoded_bytes, code_table, data_int8.shape)
        np.testing.assert_array_equal(data_int8, decoded_data)
        
        encoded_bytes, code_table = self.encoder.encode(data_int16)
        decoded_data = self.encoder.decode(encoded_bytes, code_table, data_int16.shape)
        np.testing.assert_array_equal(data_int16, decoded_data)
    
    def test_negative_values(self):
        """Test encoding data with negative values."""
        data = np.array([-5, -1, 0, 1, 5])
        
        encoded_bytes, code_table = self.encoder.encode(data)
        decoded_data = self.encoder.decode(encoded_bytes, code_table, data.shape)
        
        np.testing.assert_array_equal(data, decoded_data)
    
    def test_huffman_optimality(self):
        """Test that Huffman coding produces optimal results for known cases."""
        # Create data where we know the optimal encoding
        # Symbol frequencies: A=8, B=3, C=1 (from classic example)
        data = np.array([0]*8 + [1]*3 + [2]*1)
        
        encoded_bytes, code_table = self.encoder.encode(data)
        
        # Most frequent symbol (0) should have shortest code
        code_lengths = {symbol: len(code) for symbol, code in code_table.items()}
        
        # Symbol 0 (freq=8) should have shorter or equal code than others
        self.assertLessEqual(code_lengths[0], code_lengths[1])
        self.assertLessEqual(code_lengths[0], code_lengths[2])
        
        # Least frequent symbol (2) should have longest or equal code
        self.assertGreaterEqual(code_lengths[2], code_lengths[0])
        self.assertGreaterEqual(code_lengths[2], code_lengths[1])


if __name__ == '__main__':
    unittest.main()