"""
Integration tests for the complete compression pipeline.

These tests verify the end-to-end functionality of the compression pipeline
with different data types, parameter combinations, and edge cases.
"""

import unittest
import numpy as np
import time
from compression_pipeline import CompressionPipeline
from compression_pipeline.models import CompressedData, CompressionMetrics


class TestCompressionPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete compression pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = CompressionPipeline()
        
        # Create test datasets of different types and sizes
        self.test_datasets = {
            'small_2d': np.random.rand(8, 8).astype(np.float32),
            'medium_2d': np.random.rand(32, 32).astype(np.float32),
            'large_2d': np.random.rand(64, 64).astype(np.float32),
            'rectangular': np.random.rand(16, 24).astype(np.float32),
            'integer_data': np.random.randint(0, 255, size=(16, 16)).astype(np.uint8),
            'float_data': np.random.rand(16, 16).astype(np.float64),
            'text_data': "This is a test string for compression. It contains various characters and should compress reasonably well due to repeated patterns.",
            'binary_data': bytes(range(256)) * 4,  # 1KB of binary data
            'list_data': [[i + j for j in range(8)] for i in range(8)],
            'tuple_data': tuple(range(64)),
            'scalar_int': 42,
            'scalar_float': 3.14159
        }
    
    def test_complete_pipeline_all_data_types(self):
        """Test complete compression/decompression cycle for all supported data types."""
        for data_name, data in self.test_datasets.items():
            with self.subTest(data_type=data_name):
                # Compress data
                compressed_data = self.pipeline.compress(data)
                
                # Verify compressed data structure
                self.assertIsInstance(compressed_data, CompressedData)
                self.assertIsInstance(compressed_data.encoded_data, bytes)
                self.assertGreater(len(compressed_data.encoded_data), 0)
                self.assertIsInstance(compressed_data.huffman_table, dict)
                self.assertGreater(len(compressed_data.huffman_table), 0)
                
                # Decompress data
                reconstructed_data = self.pipeline.decompress(compressed_data)
                
                # Verify reconstruction
                self.assertIsInstance(reconstructed_data, np.ndarray)
                
                # For numeric data, check reconstruction quality
                if isinstance(data, np.ndarray):
                    self.assertEqual(reconstructed_data.shape, data.shape)
                    # Allow some reconstruction error due to lossy compression
                    mse = np.mean((data.astype(np.float64) - reconstructed_data.astype(np.float64)) ** 2)
                    self.assertLess(mse, 1000)  # Reasonable error threshold
    
    def test_compression_with_metrics(self):
        """Test compression with comprehensive metrics calculation."""
        for data_name, data in self.test_datasets.items():
            if isinstance(data, np.ndarray) and data.size > 1:  # Skip scalars for this test
                with self.subTest(data_type=data_name):
                    # Compress with metrics
                    compressed_data, metrics = self.pipeline.compress_and_measure(data)
                    
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
                    
                    # Verify compression achieved some space savings
                    self.assertLessEqual(metrics.compressed_size, metrics.original_size)
    
    def test_different_quality_settings(self):
        """Test pipeline with different quality settings."""
        test_data = self.test_datasets['medium_2d']
        quality_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = []
        
        for quality in quality_levels:
            pipeline = CompressionPipeline(quality=quality)
            compressed_data, metrics = pipeline.compress_and_measure(test_data)
            results.append((quality, metrics))
        
        # Verify quality trends
        for i in range(len(results) - 1):
            current_quality, current_metrics = results[i]
            next_quality, next_metrics = results[i + 1]
            
            # Higher quality should generally have:
            # - Lower MSE (better reconstruction)
            # - Higher PSNR (better signal quality)
            # - Higher SSIM (better structural similarity)
            # - Potentially larger compressed size
            
            if next_quality > current_quality:
                self.assertLessEqual(next_metrics.mse, current_metrics.mse * 2)  # Allow some variance
                self.assertGreaterEqual(next_metrics.psnr, current_metrics.psnr - 5)  # Allow some variance
    
    def test_different_block_sizes(self):
        """Test pipeline with different block sizes."""
        test_data = self.test_datasets['medium_2d']
        block_sizes = [4, 8, 16]
        
        for block_size in block_sizes:
            with self.subTest(block_size=block_size):
                pipeline = CompressionPipeline(block_size=block_size)
                
                # Test compression/decompression
                compressed_data = pipeline.compress(test_data)
                reconstructed_data = pipeline.decompress(compressed_data)
                
                # Verify reconstruction
                self.assertEqual(reconstructed_data.shape, test_data.shape)
                
                # Check that block size is preserved in metadata
                self.assertEqual(compressed_data.block_size, block_size)
    
    def test_different_normalization_methods(self):
        """Test pipeline with different normalization methods."""
        test_data = self.test_datasets['medium_2d'] * 1000  # Scale up for testing
        normalization_methods = ['minmax', 'zscore', 'none']
        
        for method in normalization_methods:
            with self.subTest(normalization=method):
                pipeline = CompressionPipeline(normalization=method)
                
                # Test compression/decompression
                compressed_data, metrics = pipeline.compress_and_measure(test_data)
                reconstructed_data = pipeline.decompress(compressed_data)
                
                # Verify reconstruction
                self.assertEqual(reconstructed_data.shape, test_data.shape)
                self.assertGreater(metrics.compression_ratio, 0)
                
                # Check that normalization method is preserved
                self.assertEqual(compressed_data.metadata['normalization_method'], method)
    
    def test_edge_cases_and_boundary_conditions(self):
        """Test pipeline with edge cases and boundary conditions."""
        edge_cases = {
            'single_pixel': np.array([[42]]),
            'single_row': np.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
            'single_column': np.array([[1], [2], [3], [4], [5], [6], [7], [8]]),
            'constant_data': np.ones((16, 16)),
            'zero_data': np.zeros((16, 16)),
            'high_frequency': np.random.rand(16, 16) * 2 - 1,  # [-1, 1] range
            'large_values': np.random.rand(16, 16) * 10000,
            'small_values': np.random.rand(16, 16) * 0.001,
            'non_square': np.random.rand(12, 20)
        }
        
        for case_name, data in edge_cases.items():
            with self.subTest(edge_case=case_name):
                try:
                    # Test compression/decompression
                    compressed_data = self.pipeline.compress(data)
                    reconstructed_data = self.pipeline.decompress(compressed_data)
                    
                    # Verify basic properties
                    self.assertEqual(reconstructed_data.shape, data.shape)
                    
                    # For constant data, reconstruction should be very close
                    if case_name in ['constant_data', 'zero_data']:
                        mse = np.mean((data - reconstructed_data) ** 2)
                        self.assertLess(mse, 1.0)  # Very low error for constant data
                
                except Exception as e:
                    self.fail(f"Pipeline failed on edge case '{case_name}': {str(e)}")
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create batch of different sized data
        batch_data = [
            self.test_datasets['small_2d'],
            self.test_datasets['medium_2d'],
            self.test_datasets['rectangular'],
            self.test_datasets['integer_data']
        ]
        
        # Process batch
        compressed_list, batch_stats = self.pipeline.batch_compress(batch_data)
        
        # Verify batch results
        self.assertEqual(len(compressed_list), len(batch_data))
        
        for compressed_data in compressed_list:
            self.assertIsInstance(compressed_data, CompressedData)
        
        # Verify batch statistics
        self.assertIn('batch_size', batch_stats)
        self.assertEqual(batch_stats['batch_size'], len(batch_data))
        
        expected_stats = ['compression_ratio', 'compression_time', 'decompression_time', 'mse', 'ssim']
        for stat in expected_stats:
            self.assertIn(stat, batch_stats)
            self.assertIn('mean', batch_stats[stat])
            self.assertIn('std', batch_stats[stat])
            self.assertIn('min', batch_stats[stat])
            self.assertIn('max', batch_stats[stat])
    
    def test_parameter_combinations(self):
        """Test various parameter combinations."""
        test_data = self.test_datasets['medium_2d']
        
        parameter_combinations = [
            {'block_size': 4, 'quality': 0.9, 'normalization': 'minmax'},
            {'block_size': 8, 'quality': 0.5, 'normalization': 'zscore'},
            {'block_size': 16, 'quality': 0.2, 'normalization': 'none'},
            {'block_size': 8, 'quality': 0.8, 'normalization': 'minmax'}  # Default
        ]
        
        for i, params in enumerate(parameter_combinations):
            with self.subTest(combination=i):
                pipeline = CompressionPipeline(**params)
                
                # Test compression/decompression
                compressed_data, metrics = pipeline.compress_and_measure(test_data)
                reconstructed_data = pipeline.decompress(compressed_data)
                
                # Verify reconstruction
                self.assertEqual(reconstructed_data.shape, test_data.shape)
                self.assertGreater(metrics.compression_ratio, 0)
                
                # Verify parameters are preserved
                self.assertEqual(compressed_data.block_size, params['block_size'])
                self.assertEqual(compressed_data.quality, params['quality'])
                self.assertEqual(compressed_data.metadata['normalization_method'], params['normalization'])
    
    def test_data_integrity_verification(self):
        """Test data integrity across multiple compression/decompression cycles."""
        test_data = self.test_datasets['medium_2d']
        
        # Perform multiple cycles
        current_data = test_data.copy()
        accumulated_error = 0
        
        for cycle in range(3):  # Test 3 cycles
            with self.subTest(cycle=cycle):
                # Compress and decompress
                compressed_data = self.pipeline.compress(current_data)
                reconstructed_data = self.pipeline.decompress(compressed_data)
                
                # Calculate error for this cycle
                cycle_error = np.mean((current_data - reconstructed_data) ** 2)
                accumulated_error += cycle_error
                
                # Error should not grow exponentially
                self.assertLess(cycle_error, 100)  # Reasonable per-cycle error
                
                # Update data for next cycle
                current_data = reconstructed_data
        
        # Total accumulated error should be reasonable
        self.assertLess(accumulated_error, 300)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for different data sizes."""
        data_sizes = [
            ('small', (16, 16)),
            ('medium', (64, 64)),
            ('large', (128, 128))
        ]
        
        performance_results = {}
        
        for size_name, shape in data_sizes:
            test_data = np.random.rand(*shape).astype(np.float32)
            
            # Measure compression time
            start_time = time.perf_counter()
            compressed_data, metrics = self.pipeline.compress_and_measure(test_data)
            total_time = time.perf_counter() - start_time
            
            performance_results[size_name] = {
                'data_size': test_data.nbytes,
                'compression_ratio': metrics.compression_ratio,
                'compression_time': metrics.compression_time,
                'decompression_time': metrics.decompression_time,
                'total_time': total_time,
                'throughput_mb_per_sec': (test_data.nbytes / (1024 * 1024)) / total_time
            }
        
        # Verify reasonable performance
        for size_name, results in performance_results.items():
            with self.subTest(size=size_name):
                # Compression should achieve some space savings
                self.assertGreater(results['compression_ratio'], 1.0)
                
                # Times should be reasonable (less than 10 seconds for test data)
                self.assertLess(results['total_time'], 10.0)
                
                # Throughput should be reasonable (at least 0.1 MB/s)
                self.assertGreater(results['throughput_mb_per_sec'], 0.1)
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during compression."""
        # This test verifies that the pipeline doesn't have major memory leaks
        import gc
        
        test_data = self.test_datasets['large_2d']
        
        # Perform multiple compressions and check memory doesn't grow excessively
        initial_objects = len(gc.get_objects())
        
        for i in range(10):
            compressed_data = self.pipeline.compress(test_data)
            reconstructed_data = self.pipeline.decompress(compressed_data)
            
            # Force garbage collection
            gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow excessively (allow some growth for caching, etc.)
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable threshold
    
    def test_concurrent_pipeline_usage(self):
        """Test using multiple pipeline instances concurrently."""
        import threading
        
        test_data = self.test_datasets['medium_2d']
        results = {}
        errors = {}
        
        def compress_decompress(pipeline_id):
            try:
                pipeline = CompressionPipeline(quality=0.5 + pipeline_id * 0.1)
                compressed_data, metrics = pipeline.compress_and_measure(test_data)
                reconstructed_data = pipeline.decompress(compressed_data)
                results[pipeline_id] = {
                    'metrics': metrics,
                    'reconstruction_error': np.mean((test_data - reconstructed_data) ** 2)
                }
            except Exception as e:
                errors[pipeline_id] = str(e)
        
        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=compress_decompress, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 3)
        
        for pipeline_id, result in results.items():
            self.assertGreater(result['metrics'].compression_ratio, 0)
            self.assertLess(result['reconstruction_error'], 100)


if __name__ == '__main__':
    unittest.main()