"""
Unit tests for main compression pipeline.
"""

import unittest
import numpy as np
from compression_pipeline.pipeline import CompressionPipeline
from compression_pipeline.models import CompressedData, CompressionMetrics


class TestCompressionPipeline(unittest.TestCase):
    """Test cases for CompressionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = CompressionPipeline()
        self.high_quality_pipeline = CompressionPipeline(quality=0.9)
        self.low_quality_pipeline = CompressionPipeline(quality=0.3)
    
    def test_initialization(self):
        """Test CompressionPipeline initialization."""
        # Default initialization
        pipeline = CompressionPipeline()
        self.assertEqual(pipeline.block_size, 8)
        self.assertEqual(pipeline.quality, 0.8)
        self.assertEqual(pipeline.normalization, 'minmax')
        
        # Custom initialization
        custom_pipeline = CompressionPipeline(block_size=4, quality=0.5, normalization='zscore')
        self.assertEqual(custom_pipeline.block_size, 4)
        self.assertEqual(custom_pipeline.quality, 0.5)
        self.assertEqual(custom_pipeline.normalization, 'zscore')
    
    def test_invalid_initialization(self):
        """Test error handling for invalid initialization parameters."""
        # Invalid block size
        with self.assertRaises(ValueError):
            CompressionPipeline(block_size=0)
        
        # Invalid quality
        with self.assertRaises(ValueError):
            CompressionPipeline(quality=0.0)
        
        with self.assertRaises(ValueError):
            CompressionPipeline(quality=1.1)
        
        # Invalid normalization
        with self.assertRaises(ValueError):
            CompressionPipeline(normalization='invalid')
    
    def test_compress_decompress_numpy_array(self):
        """Test compression and decompression of numpy arrays."""
        # Create test data
        original_data = np.random.rand(16, 16).astype(np.float32)
        
        # Compress
        compressed_data = self.pipeline.compress(original_data)
        
        # Verify compressed data structure
        self.assertIsInstance(compressed_data, CompressedData)
        self.assertIsInstance(compressed_data.encoded_data, bytes)
        self.assertIsInstance(compressed_data.huffman_table, dict)
        self.assertEqual(compressed_data.original_shape, original_data.shape)
        
        # Decompress
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        # Verify reconstruction
        self.assertEqual(reconstructed_data.shape, original_data.shape)
        
        # Check reconstruction quality (should be reasonably close)
        mse = np.mean((original_data - reconstructed_data) ** 2)
        self.assertLess(mse, 100)  # Reasonable error threshold
    
    def test_compress_decompress_list(self):
        """Test compression and decompression of list data."""
        original_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        
        compressed_data = self.pipeline.compress(original_data)
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        # Should reconstruct to similar values
        original_array = np.array(original_data)
        np.testing.assert_allclose(original_array, reconstructed_data, rtol=0.1)
    
    def test_compress_decompress_text(self):
        """Test compression and decompression of text data."""
        original_text = "Hello, World! This is a test string for compression."
        
        compressed_data = self.pipeline.compress(original_text)
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        # For text, we expect some loss due to the lossy compression
        # But the general structure should be preserved
        self.assertEqual(reconstructed_data.shape, (len(original_text.encode('utf-8')),))
    
    def test_compress_decompress_binary(self):
        """Test compression and decompression of binary data."""
        original_binary = b"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07" * 8
        
        compressed_data = self.pipeline.compress(original_binary)
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        self.assertEqual(reconstructed_data.shape, (len(original_binary),))
    
    def test_quality_impact(self):
        """Test that quality parameter affects compression results."""
        original_data = np.random.rand(32, 32).astype(np.float32)
        
        # Compress with different quality settings
        high_quality_compressed = self.high_quality_pipeline.compress(original_data)
        low_quality_compressed = self.low_quality_pipeline.compress(original_data)
        
        # Decompress both
        high_quality_reconstructed = self.high_quality_pipeline.decompress(high_quality_compressed)
        low_quality_reconstructed = self.low_quality_pipeline.decompress(low_quality_compressed)
        
        # Calculate reconstruction errors
        high_quality_mse = np.mean((original_data - high_quality_reconstructed) ** 2)
        low_quality_mse = np.mean((original_data - low_quality_reconstructed) ** 2)
        
        # High quality should have lower reconstruction error
        self.assertLess(high_quality_mse, low_quality_mse)
        
        # Low quality should achieve better compression (smaller size)
        self.assertLessEqual(len(low_quality_compressed.encoded_data), 
                           len(high_quality_compressed.encoded_data))
    
    def test_compress_and_measure(self):
        """Test compression with metrics calculation."""
        original_data = np.random.rand(16, 16).astype(np.float32)
        
        compressed_data, metrics = self.pipeline.compress_and_measure(original_data)
        
        # Verify compressed data
        self.assertIsInstance(compressed_data, CompressedData)
        
        # Verify metrics
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreater(metrics.compression_ratio, 0)
        self.assertGreater(metrics.original_size, 0)
        self.assertGreater(metrics.compressed_size, 0)
        self.assertGreaterEqual(metrics.compression_time, 0)
        self.assertGreaterEqual(metrics.decompression_time, 0)
        self.assertGreaterEqual(metrics.mse, 0)
        self.assertGreater(metrics.psnr, 0)
        self.assertGreaterEqual(metrics.ssim, -1)
        self.assertLessEqual(metrics.ssim, 1)
        
        # Check that metrics are stored
        stored_metrics = self.pipeline.get_metrics()
        self.assertEqual(stored_metrics, metrics)
    
    def test_different_block_sizes(self):
        """Test pipeline with different block sizes."""
        original_data = np.random.rand(24, 24).astype(np.float32)
        
        # Test with 4x4 blocks
        pipeline_4x4 = CompressionPipeline(block_size=4)
        compressed_4x4 = pipeline_4x4.compress(original_data)
        reconstructed_4x4 = pipeline_4x4.decompress(compressed_4x4)
        
        self.assertEqual(reconstructed_4x4.shape, original_data.shape)
        
        # Test with 16x16 blocks
        pipeline_16x16 = CompressionPipeline(block_size=16)
        compressed_16x16 = pipeline_16x16.compress(original_data)
        reconstructed_16x16 = pipeline_16x16.decompress(compressed_16x16)
        
        self.assertEqual(reconstructed_16x16.shape, original_data.shape)
    
    def test_different_normalization_methods(self):
        """Test pipeline with different normalization methods."""
        original_data = np.random.rand(16, 16) * 1000  # Large range
        
        # Test minmax normalization
        pipeline_minmax = CompressionPipeline(normalization='minmax')
        compressed_minmax = pipeline_minmax.compress(original_data)
        reconstructed_minmax = pipeline_minmax.decompress(compressed_minmax)
        
        # Test zscore normalization
        pipeline_zscore = CompressionPipeline(normalization='zscore')
        compressed_zscore = pipeline_zscore.compress(original_data)
        reconstructed_zscore = pipeline_zscore.decompress(compressed_zscore)
        
        # Test no normalization
        pipeline_none = CompressionPipeline(normalization='none')
        compressed_none = pipeline_none.compress(original_data)
        reconstructed_none = pipeline_none.decompress(compressed_none)
        
        # All should reconstruct to similar shapes
        self.assertEqual(reconstructed_minmax.shape, original_data.shape)
        self.assertEqual(reconstructed_zscore.shape, original_data.shape)
        self.assertEqual(reconstructed_none.shape, original_data.shape)
    
    def test_set_quality(self):
        """Test dynamic quality adjustment."""
        original_quality = self.pipeline.quality
        
        # Change quality
        new_quality = 0.5
        self.pipeline.set_quality(new_quality)
        
        self.assertEqual(self.pipeline.quality, new_quality)
        self.assertNotEqual(self.pipeline.quality, original_quality)
        
        # Test invalid quality
        with self.assertRaises(ValueError):
            self.pipeline.set_quality(0.0)
    
    def test_set_block_size(self):
        """Test dynamic block size adjustment."""
        original_block_size = self.pipeline.block_size
        
        # Change block size
        new_block_size = 4
        self.pipeline.set_block_size(new_block_size)
        
        self.assertEqual(self.pipeline.block_size, new_block_size)
        self.assertNotEqual(self.pipeline.block_size, original_block_size)
        
        # Test invalid block size
        with self.assertRaises(ValueError):
            self.pipeline.set_block_size(0)
    
    def test_pipeline_info(self):
        """Test pipeline information retrieval."""
        info = self.pipeline.get_pipeline_info()
        
        expected_keys = ['block_size', 'quality', 'normalization', 
                        'quantization_table_shape', 'components']
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['block_size'], 8)
        self.assertEqual(info['quality'], 0.8)
        self.assertEqual(info['normalization'], 'minmax')
        
        # Check components
        components = info['components']
        expected_components = ['dct_processor', 'quantizer', 'huffman_encoder', 
                             'data_preprocessor', 'metrics_collector']
        
        for component in expected_components:
            self.assertIn(component, components)
    
    def test_batch_compress(self):
        """Test batch compression functionality."""
        # Create multiple data items
        data_list = [
            np.random.rand(8, 8),
            np.random.rand(12, 12),
            np.random.rand(16, 16)
        ]
        
        compressed_list, batch_stats = self.pipeline.batch_compress(data_list)
        
        # Check compressed list
        self.assertEqual(len(compressed_list), len(data_list))
        for compressed_data in compressed_list:
            self.assertIsInstance(compressed_data, CompressedData)
        
        # Check batch statistics
        self.assertIn('batch_size', batch_stats)
        self.assertEqual(batch_stats['batch_size'], 3)
        
        expected_stat_keys = ['compression_ratio', 'compression_time', 
                            'decompression_time', 'mse', 'ssim']
        
        for key in expected_stat_keys:
            self.assertIn(key, batch_stats)
            self.assertIn('mean', batch_stats[key])
            self.assertIn('std', batch_stats[key])
            self.assertIn('min', batch_stats[key])
            self.assertIn('max', batch_stats[key])
    
    def test_compressed_data_validation(self):
        """Test validation of compressed data during decompression."""
        original_data = np.random.rand(8, 8)
        compressed_data = self.pipeline.compress(original_data)
        
        # Test with wrong block size
        wrong_pipeline = CompressionPipeline(block_size=4)
        with self.assertRaises(ValueError):
            wrong_pipeline.decompress(compressed_data)
        
        # Test with corrupted metadata
        corrupted_data = CompressedData(
            encoded_data=compressed_data.encoded_data,
            huffman_table=compressed_data.huffman_table,
            original_shape=compressed_data.original_shape,
            block_size=compressed_data.block_size,
            quality=compressed_data.quality,
            quantization_table=compressed_data.quantization_table,
            metadata={}  # Missing required metadata
        )
        
        with self.assertRaises(ValueError):
            self.pipeline.decompress(corrupted_data)
    
    def test_empty_data_handling(self):
        """Test error handling for empty or invalid data."""
        # Empty array
        with self.assertRaises(ValueError):
            self.pipeline.compress(np.array([]))
        
        # None data
        with self.assertRaises(ValueError):
            self.pipeline.compress(None)
    
    def test_large_data_compression(self):
        """Test compression of larger datasets."""
        # Create larger test data
        large_data = np.random.rand(64, 64).astype(np.float32)
        
        compressed_data, metrics = self.pipeline.compress_and_measure(large_data)
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        # Should handle large data correctly
        self.assertEqual(reconstructed_data.shape, large_data.shape)
        self.assertGreater(metrics.compression_ratio, 0)
        
        # Should achieve some compression
        self.assertLess(metrics.compressed_size, metrics.original_size)


if __name__ == '__main__':
    unittest.main()