"""
Unit tests for DCT processor component.
"""

import unittest
import numpy as np
from compression_pipeline.dct_processor import DCTProcessor


class TestDCTProcessor(unittest.TestCase):
    """Test cases for DCTProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DCTProcessor(block_size=8)
        self.small_processor = DCTProcessor(block_size=4)
    
    def test_initialization(self):
        """Test DCTProcessor initialization."""
        # Valid initialization
        processor = DCTProcessor(8)
        self.assertEqual(processor.block_size, 8)
        
        # Invalid block size
        with self.assertRaises(ValueError):
            DCTProcessor(0)
        
        with self.assertRaises(ValueError):
            DCTProcessor(-1)
    
    def test_forward_dct_basic(self):
        """Test basic forward DCT functionality."""
        # Create a simple 8x8 test block
        test_block = np.random.rand(8, 8).astype(np.float32)
        
        # Apply DCT
        dct_coeffs = self.processor.forward_dct(test_block)
        
        # Check output properties
        self.assertEqual(dct_coeffs.shape, (8, 8))
        self.assertEqual(dct_coeffs.dtype, np.float32)
        
        # DC coefficient should be non-zero for non-zero input
        self.assertNotEqual(dct_coeffs[0, 0], 0)
    
    def test_inverse_dct_basic(self):
        """Test basic inverse DCT functionality."""
        # Create DCT coefficients
        dct_coeffs = np.random.rand(8, 8).astype(np.float32)
        
        # Apply inverse DCT
        reconstructed = self.processor.inverse_dct(dct_coeffs)
        
        # Check output properties
        self.assertEqual(reconstructed.shape, (8, 8))
        self.assertEqual(reconstructed.dtype, np.float32)
    
    def test_dct_inverse_dct_roundtrip(self):
        """Test that DCT followed by inverse DCT reconstructs original data."""
        # Create test data
        original = np.random.rand(8, 8).astype(np.float32)
        
        # Apply DCT then inverse DCT
        dct_coeffs = self.processor.forward_dct(original)
        reconstructed = self.processor.inverse_dct(dct_coeffs)
        
        # Check reconstruction accuracy (should be very close due to floating point precision)
        np.testing.assert_allclose(original, reconstructed, rtol=1e-5, atol=1e-6)
    
    def test_block_processing_no_padding(self):
        """Test block processing when no padding is needed."""
        # Create data that's exactly divisible by block size
        data = np.random.rand(16, 24).astype(np.float32)
        
        # Process forward then inverse
        dct_result = self.processor.process_blocks(data, inverse=False)
        reconstructed = self.processor.process_blocks(dct_result, inverse=True)
        
        # Check shape preservation and reconstruction accuracy
        self.assertEqual(reconstructed.shape, data.shape)
        np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-5)
    
    def test_block_processing_with_padding(self):
        """Test block processing when padding is needed."""
        # Create data that requires padding
        data = np.random.rand(10, 15).astype(np.float32)
        
        # Process forward then inverse
        dct_result = self.processor.process_blocks(data, inverse=False)
        reconstructed = self.processor.process_blocks(dct_result, inverse=True)
        
        # Check shape preservation and reconstruction accuracy
        self.assertEqual(reconstructed.shape, data.shape)
        np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-5)
    
    def test_different_block_sizes(self):
        """Test DCT processing with different block sizes."""
        data = np.random.rand(12, 12).astype(np.float32)
        
        # Test with 4x4 blocks
        dct_result = self.small_processor.process_blocks(data, inverse=False)
        reconstructed = self.small_processor.process_blocks(dct_result, inverse=True)
        
        self.assertEqual(reconstructed.shape, data.shape)
        np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-5)
    
    def test_invalid_block_shapes(self):
        """Test error handling for invalid block shapes."""
        # Wrong shape for forward DCT
        with self.assertRaises(ValueError):
            self.processor.forward_dct(np.random.rand(4, 4))
        
        # Wrong shape for inverse DCT
        with self.assertRaises(ValueError):
            self.processor.inverse_dct(np.random.rand(4, 4))
    
    def test_invalid_data_dimensions(self):
        """Test error handling for invalid data dimensions."""
        # 1D data
        with self.assertRaises(ValueError):
            self.processor.process_blocks(np.random.rand(10))
        
        # 3D data
        with self.assertRaises(ValueError):
            self.processor.process_blocks(np.random.rand(10, 10, 3))
    
    def test_dct_energy_compaction(self):
        """Test that DCT provides energy compaction (most energy in low frequencies)."""
        # Create a smooth test signal (should have energy concentrated in low frequencies)
        x, y = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        smooth_signal = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)
        
        dct_coeffs = self.processor.forward_dct(smooth_signal)
        
        # Energy should be concentrated in the top-left (low frequency) region
        low_freq_energy = np.sum(dct_coeffs[:4, :4] ** 2)
        total_energy = np.sum(dct_coeffs ** 2)
        
        # At least 50% of energy should be in low frequencies for smooth signals
        self.assertGreater(low_freq_energy / total_energy, 0.5)


if __name__ == '__main__':
    unittest.main()