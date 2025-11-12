"""
Unit tests for quantization component.
"""

import unittest
import numpy as np
from compression_pipeline.quantizer import Quantizer


class TestQuantizer(unittest.TestCase):
    """Test cases for Quantizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quantizer = Quantizer(quality=0.8)
        self.low_quality = Quantizer(quality=0.2)
        self.high_quality = Quantizer(quality=0.95)
    
    def test_initialization(self):
        """Test Quantizer initialization."""
        # Valid initialization
        quantizer = Quantizer(0.5)
        self.assertEqual(quantizer.quality, 0.5)
        self.assertEqual(quantizer.block_size, 8)
        
        # Invalid quality values
        with self.assertRaises(ValueError):
            Quantizer(0.0)
        
        with self.assertRaises(ValueError):
            Quantizer(1.1)
        
        with self.assertRaises(ValueError):
            Quantizer(-0.1)
    
    def test_quantization_table_generation(self):
        """Test quantization table generation."""
        table = self.quantizer.get_quantization_table()
        
        # Check table properties
        self.assertEqual(table.shape, (8, 8))
        self.assertTrue(np.all(table > 0))  # All values should be positive
        
        # DC component should be smallest (least quantization)
        self.assertEqual(table[0, 0], np.min(table))
        
        # Higher quality should have smaller quantization values
        high_q_table = self.high_quality.get_quantization_table()
        low_q_table = self.low_quality.get_quantization_table()
        
        # High quality table should have smaller values (less quantization)
        self.assertTrue(np.all(high_q_table <= low_q_table))
    
    def test_quantize_basic(self):
        """Test basic quantization functionality."""
        # Create test DCT coefficients
        dct_coeffs = np.random.rand(8, 8).astype(np.float32) * 100
        
        # Apply quantization
        quantized = self.quantizer.quantize(dct_coeffs)
        
        # Check output properties
        self.assertEqual(quantized.shape, (8, 8))
        self.assertEqual(quantized.dtype, np.int16)
        
        # Quantized values should be integers
        self.assertTrue(np.all(quantized == quantized.astype(int)))
    
    def test_dequantize_basic(self):
        """Test basic dequantization functionality."""
        # Create test quantized coefficients
        quantized = np.random.randint(-100, 100, size=(8, 8)).astype(np.int16)
        
        # Apply dequantization
        dequantized = self.quantizer.dequantize(quantized)
        
        # Check output properties
        self.assertEqual(dequantized.shape, (8, 8))
        self.assertEqual(dequantized.dtype, np.float32)
    
    def test_quantize_dequantize_roundtrip(self):
        """Test quantization followed by dequantization."""
        # Create test DCT coefficients
        original = np.random.rand(8, 8).astype(np.float32) * 50
        
        # Apply quantization then dequantization
        quantized = self.quantizer.quantize(original)
        dequantized = self.quantizer.dequantize(quantized)
        
        # Check that dequantized values are reasonably close to original
        # (some loss is expected due to quantization)
        mse = np.mean((original - dequantized) ** 2)
        self.assertLess(mse, 100)  # Reasonable error threshold
    
    def test_quality_impact_on_quantization(self):
        """Test that quality parameter affects quantization results."""
        # Create test coefficients
        dct_coeffs = np.random.rand(8, 8).astype(np.float32) * 100
        
        # Quantize with different quality settings
        high_q_quantized = self.high_quality.quantize(dct_coeffs)
        low_q_quantized = self.low_quality.quantize(dct_coeffs)
        
        # Dequantize both
        high_q_dequantized = self.high_quality.dequantize(high_q_quantized)
        low_q_dequantized = self.low_quality.dequantize(low_q_quantized)
        
        # High quality should have lower reconstruction error
        high_q_error = np.mean((dct_coeffs - high_q_dequantized) ** 2)
        low_q_error = np.mean((dct_coeffs - low_q_dequantized) ** 2)
        
        self.assertLess(high_q_error, low_q_error)
    
    def test_custom_quantization_table(self):
        """Test setting custom quantization table."""
        # Create custom table
        custom_table = np.ones((8, 8), dtype=np.float32) * 10
        
        # Set custom table
        self.quantizer.set_custom_quantization_table(custom_table)
        
        # Verify table was set
        retrieved_table = self.quantizer.get_quantization_table()
        np.testing.assert_array_equal(custom_table, retrieved_table)
        
        # Test invalid custom tables
        with self.assertRaises(ValueError):
            # Wrong shape
            self.quantizer.set_custom_quantization_table(np.ones((4, 4)))
        
        with self.assertRaises(ValueError):
            # Negative values
            self.quantizer.set_custom_quantization_table(np.ones((8, 8)) * -1)
        
        with self.assertRaises(ValueError):
            # Zero values
            self.quantizer.set_custom_quantization_table(np.zeros((8, 8)))
    
    def test_different_block_sizes(self):
        """Test quantizer with different block sizes."""
        # Test 4x4 quantizer
        quantizer_4x4 = Quantizer(quality=0.8, block_size=4)
        
        table = quantizer_4x4.get_quantization_table()
        self.assertEqual(table.shape, (4, 4))
        
        # Test quantization with 4x4 blocks
        dct_coeffs = np.random.rand(4, 4).astype(np.float32) * 50
        quantized = quantizer_4x4.quantize(dct_coeffs)
        dequantized = quantizer_4x4.dequantize(quantized)
        
        self.assertEqual(quantized.shape, (4, 4))
        self.assertEqual(dequantized.shape, (4, 4))
    
    def test_invalid_coefficient_shapes(self):
        """Test error handling for invalid coefficient shapes."""
        # Wrong shape for quantization
        with self.assertRaises(ValueError):
            self.quantizer.quantize(np.random.rand(4, 4))
        
        # Wrong shape for dequantization
        with self.assertRaises(ValueError):
            self.quantizer.dequantize(np.random.randint(0, 10, size=(4, 4)))
    
    def test_quantization_error_calculation(self):
        """Test quantization error calculation."""
        # Create test coefficients
        original = np.random.rand(8, 8).astype(np.float32) * 100
        quantized = self.quantizer.quantize(original)
        
        # Calculate error
        error = self.quantizer.calculate_quantization_error(original, quantized)
        
        # Error should be non-negative
        self.assertGreaterEqual(error, 0)
        
        # Error should be reasonable (not too large)
        self.assertLess(error, 10000)
    
    def test_dc_coefficient_preservation(self):
        """Test that DC coefficient is preserved reasonably well."""
        # Create coefficients with strong DC component
        dct_coeffs = np.zeros((8, 8), dtype=np.float32)
        dct_coeffs[0, 0] = 100  # Strong DC component
        
        # Quantize and dequantize
        quantized = self.quantizer.quantize(dct_coeffs)
        dequantized = self.quantizer.dequantize(quantized)
        
        # DC component should be preserved reasonably well
        dc_error = abs(dct_coeffs[0, 0] - dequantized[0, 0])
        self.assertLess(dc_error, 20)  # Allow some error but not too much
    
    def test_quantization_range_clipping(self):
        """Test that quantization properly clips extreme values."""
        # Create coefficients with extreme values
        dct_coeffs = np.full((8, 8), 1e6, dtype=np.float32)
        
        # Quantize (should clip to int16 range)
        quantized = self.quantizer.quantize(dct_coeffs)
        
        # Check that values are within int16 range
        self.assertTrue(np.all(quantized >= -32768))
        self.assertTrue(np.all(quantized <= 32767))


if __name__ == '__main__':
    unittest.main()