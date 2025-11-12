"""
Unit tests for metrics collection system.
"""

import unittest
import numpy as np
import time
from compression_pipeline.metrics_collector import MetricsCollector
from compression_pipeline.models import CompressionMetrics


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
    
    def test_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        self.assertEqual(len(collector.batch_metrics), 0)
        self.assertIsNone(collector.current_compression_start)
        self.assertIsNone(collector.current_decompression_start)
    
    def test_compression_timing(self):
        """Test compression timing functionality."""
        self.collector.start_compression_timer()
        time.sleep(0.01)  # Small delay
        elapsed = self.collector.end_compression_timer()
        
        self.assertGreater(elapsed, 0)
        self.assertLess(elapsed, 1)  # Should be much less than 1 second
    
    def test_decompression_timing(self):
        """Test decompression timing functionality."""
        self.collector.start_decompression_timer()
        time.sleep(0.01)  # Small delay
        elapsed = self.collector.end_decompression_timer()
        
        self.assertGreater(elapsed, 0)
        self.assertLess(elapsed, 1)  # Should be much less than 1 second
    
    def test_timing_errors(self):
        """Test error handling for timing operations."""
        # End timer without starting
        with self.assertRaises(RuntimeError):
            self.collector.end_compression_timer()
        
        with self.assertRaises(RuntimeError):
            self.collector.end_decompression_timer()
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        # Normal case
        ratio = self.collector.calculate_compression_ratio(1000, 500)
        self.assertEqual(ratio, 2.0)
        
        # Perfect compression (impossible but test edge case)
        ratio = self.collector.calculate_compression_ratio(1000, 0)
        self.assertEqual(ratio, float('inf'))
        
        # No compression
        ratio = self.collector.calculate_compression_ratio(1000, 1000)
        self.assertEqual(ratio, 1.0)
    
    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        original = np.array([1.0, 2.0, 3.0, 4.0])
        reconstructed = np.array([1.1, 2.1, 2.9, 3.9])
        
        mse = self.collector.calculate_mse(original, reconstructed)
        expected_mse = np.mean([0.01, 0.01, 0.01, 0.01])
        
        self.assertAlmostEqual(mse, expected_mse, places=6)
    
    def test_mse_perfect_reconstruction(self):
        """Test MSE with perfect reconstruction."""
        original = np.array([1.0, 2.0, 3.0, 4.0])
        reconstructed = original.copy()
        
        mse = self.collector.calculate_mse(original, reconstructed)
        self.assertEqual(mse, 0.0)
    
    def test_mse_shape_mismatch(self):
        """Test MSE error handling for shape mismatch."""
        original = np.array([1, 2, 3])
        reconstructed = np.array([1, 2])
        
        with self.assertRaises(ValueError):
            self.collector.calculate_mse(original, reconstructed)
    
    def test_psnr_calculation(self):
        """Test PSNR calculation."""
        original = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
        reconstructed = np.array([5, 55, 95, 155, 195, 250], dtype=np.uint8)
        
        psnr = self.collector.calculate_psnr(original, reconstructed, max_value=255)
        
        # PSNR should be positive for reasonable reconstruction
        self.assertGreater(psnr, 0)
        self.assertLess(psnr, 100)  # Reasonable upper bound
    
    def test_psnr_perfect_reconstruction(self):
        """Test PSNR with perfect reconstruction."""
        original = np.array([1, 2, 3, 4])
        reconstructed = original.copy()
        
        psnr = self.collector.calculate_psnr(original, reconstructed)
        self.assertEqual(psnr, float('inf'))
    
    def test_psnr_auto_max_value(self):
        """Test PSNR with automatic max value detection."""
        original = np.array([0.0, 0.5, 1.0])
        reconstructed = np.array([0.1, 0.4, 0.9])
        
        psnr = self.collector.calculate_psnr(original, reconstructed)
        self.assertGreater(psnr, 0)
    
    def test_ssim_calculation_2d(self):
        """Test SSIM calculation for 2D arrays."""
        # Create simple 2D test data
        original = np.random.rand(8, 8)
        # Add small noise for reconstruction
        reconstructed = original + np.random.normal(0, 0.01, original.shape)
        
        ssim_value = self.collector.calculate_ssim(original, reconstructed)
        
        # SSIM should be between -1 and 1, and close to 1 for similar images
        self.assertGreaterEqual(ssim_value, -1)
        self.assertLessEqual(ssim_value, 1)
        self.assertGreater(ssim_value, 0.8)  # Should be high for low noise
    
    def test_ssim_calculation_1d(self):
        """Test SSIM calculation for 1D arrays."""
        # Create 1D data that can be reshaped to square
        size = 16  # 4x4 when reshaped
        original = np.random.rand(size)
        reconstructed = original + np.random.normal(0, 0.01, size)
        
        ssim_value = self.collector.calculate_ssim(original, reconstructed)
        
        self.assertGreaterEqual(ssim_value, -1)
        self.assertLessEqual(ssim_value, 1)
    
    def test_ssim_perfect_similarity(self):
        """Test SSIM with identical arrays."""
        original = np.random.rand(8, 8)
        reconstructed = original.copy()
        
        ssim_value = self.collector.calculate_ssim(original, reconstructed)
        self.assertAlmostEqual(ssim_value, 1.0, places=5)
    
    def test_ssim_constant_arrays(self):
        """Test SSIM with constant arrays."""
        original = np.ones((8, 8))
        reconstructed = np.ones((8, 8))
        
        ssim_value = self.collector.calculate_ssim(original, reconstructed)
        self.assertEqual(ssim_value, 1.0)
    
    def test_create_metrics(self):
        """Test comprehensive metrics creation."""
        original = np.random.rand(10, 10).astype(np.float32)
        reconstructed = original + np.random.normal(0, 0.01, original.shape).astype(np.float32)
        
        metrics = self.collector.create_metrics(
            original_data=original,
            reconstructed_data=reconstructed,
            compressed_size=200,
            compression_time=0.1,
            decompression_time=0.05
        )
        
        # Check that all metrics are calculated
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreater(metrics.compression_ratio, 0)
        self.assertEqual(metrics.original_size, original.nbytes)
        self.assertEqual(metrics.compressed_size, 200)
        self.assertEqual(metrics.compression_time, 0.1)
        self.assertEqual(metrics.decompression_time, 0.05)
        self.assertGreaterEqual(metrics.mse, 0)
        self.assertGreater(metrics.psnr, 0)
        self.assertGreaterEqual(metrics.ssim, -1)
        self.assertLessEqual(metrics.ssim, 1)
    
    def test_batch_metrics_collection(self):
        """Test batch metrics collection and statistics."""
        # Create multiple metrics
        for i in range(5):
            original = np.random.rand(5, 5)
            reconstructed = original + np.random.normal(0, 0.01, original.shape)
            
            metrics = self.collector.create_metrics(
                original_data=original,
                reconstructed_data=reconstructed,
                compressed_size=50 + i * 10,
                compression_time=0.1 + i * 0.01,
                decompression_time=0.05 + i * 0.005
            )
            
            self.collector.add_batch_metrics(metrics)
        
        # Check batch size
        self.assertEqual(len(self.collector.batch_metrics), 5)
        
        # Get batch statistics
        stats = self.collector.get_batch_statistics()
        
        # Check statistics structure
        self.assertIn('batch_size', stats)
        self.assertIn('compression_ratio', stats)
        self.assertIn('compression_time', stats)
        self.assertIn('decompression_time', stats)
        self.assertIn('mse', stats)
        self.assertIn('ssim', stats)
        
        self.assertEqual(stats['batch_size'], 5)
        
        # Check that statistics contain expected keys
        for metric_name in ['compression_ratio', 'compression_time', 'decompression_time', 'mse', 'ssim']:
            metric_stats = stats[metric_name]
            self.assertIn('mean', metric_stats)
            self.assertIn('std', metric_stats)
            self.assertIn('min', metric_stats)
            self.assertIn('max', metric_stats)
    
    def test_empty_batch_statistics(self):
        """Test batch statistics with no metrics."""
        stats = self.collector.get_batch_statistics()
        self.assertEqual(stats, {})
    
    def test_clear_batch_metrics(self):
        """Test clearing batch metrics."""
        # Add some metrics
        metrics = CompressionMetrics(2.0, 100, 50, 0.1, 0.05, 1.0, 30.0, 0.9)
        self.collector.add_batch_metrics(metrics)
        
        self.assertEqual(len(self.collector.batch_metrics), 1)
        
        # Clear metrics
        self.collector.clear_batch_metrics()
        self.assertEqual(len(self.collector.batch_metrics), 0)
    
    def test_export_metrics_to_dict(self):
        """Test exporting metrics to dictionary."""
        metrics = CompressionMetrics(2.5, 1000, 400, 0.15, 0.08, 2.5, 35.2, 0.85)
        
        exported = self.collector.export_metrics_to_dict(metrics)
        
        expected_keys = [
            'compression_ratio', 'original_size_bytes', 'compressed_size_bytes',
            'compression_time_seconds', 'decompression_time_seconds',
            'mse', 'psnr_db', 'ssim', 'space_savings_percent'
        ]
        
        for key in expected_keys:
            self.assertIn(key, exported)
        
        self.assertEqual(exported['compression_ratio'], 2.5)
        self.assertEqual(exported['space_savings_percent'], 60.0)  # (1 - 1/2.5) * 100
    
    def test_compare_metrics(self):
        """Test metrics comparison."""
        metrics1 = CompressionMetrics(3.0, 1000, 333, 0.1, 0.05, 1.0, 40.0, 0.9)
        metrics2 = CompressionMetrics(2.0, 1000, 500, 0.15, 0.08, 2.0, 35.0, 0.8)
        
        comparison = self.collector.compare_metrics(metrics1, metrics2)
        
        # metrics1 should be better in most aspects
        self.assertGreater(comparison['compression_ratio_diff'], 0)  # Higher ratio is better
        self.assertGreater(comparison['compression_time_diff'], 0)  # Lower time is better (diff is positive)
        self.assertGreater(comparison['decompression_time_diff'], 0)  # Lower time is better
        self.assertGreater(comparison['mse_diff'], 0)  # Lower MSE is better (diff is positive)
        self.assertGreater(comparison['psnr_diff'], 0)  # Higher PSNR is better
        self.assertGreater(comparison['ssim_diff'], 0)  # Higher SSIM is better


if __name__ == '__main__':
    unittest.main()