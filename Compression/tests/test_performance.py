"""
Unit tests for performance optimization utilities.
"""

import unittest
import numpy as np
import time
from compression_pipeline.pipeline import CompressionPipeline
from compression_pipeline.performance import PerformanceOptimizer


class TestPerformanceOptimizer(unittest.TestCase):
    """Test cases for PerformanceOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = CompressionPipeline()
        self.optimizer = PerformanceOptimizer(self.pipeline)
    
    def test_initialization(self):
        """Test PerformanceOptimizer initialization."""
        optimizer = PerformanceOptimizer(self.pipeline)
        self.assertEqual(optimizer.pipeline, self.pipeline)
        self.assertEqual(len(optimizer.benchmark_results), 0)
    
    def test_chunk_compress_large_data(self):
        """Test chunk-based compression for large data."""
        # Create test data
        large_data = np.random.rand(100, 100).astype(np.float32)
        
        # Compress in chunks
        compressed_chunks = list(self.optimizer.chunk_compress_large_data(
            large_data, chunk_size=(32, 32), overlap=4
        ))
        
        # Verify chunks were created
        self.assertGreater(len(compressed_chunks), 1)
        
        # Verify each chunk is properly compressed
        for compressed_chunk, position in compressed_chunks:
            self.assertIsNotNone(compressed_chunk)
            self.assertIsInstance(position, tuple)
            self.assertEqual(len(position), 2)
    
    def test_reconstruct_from_chunks(self):
        """Test reconstruction from compressed chunks."""
        # Create test data
        test_data = np.random.rand(64, 64).astype(np.float32)
        
        # Compress in chunks
        compressed_chunks = list(self.optimizer.chunk_compress_large_data(
            test_data, chunk_size=(24, 24), overlap=4
        ))
        
        # Reconstruct from chunks
        reconstructed_data = self.optimizer.reconstruct_from_chunks(
            compressed_chunks, test_data.shape, chunk_size=(24, 24), overlap=4
        )
        
        # Verify reconstruction
        self.assertEqual(reconstructed_data.shape, test_data.shape)
        
        # Check reconstruction quality (allow some error due to chunking and compression)
        mse = np.mean((test_data - reconstructed_data) ** 2)
        self.assertLess(mse, 100)  # Reasonable error threshold
    
    def test_memory_efficient_batch_compress(self):
        """Test memory-efficient batch compression."""
        # Create test data list
        data_list = [
            np.random.rand(16, 16).astype(np.float32),
            np.random.rand(20, 20).astype(np.float32),
            np.random.rand(12, 12).astype(np.float32)
        ]
        
        # Process with memory limit
        results = list(self.optimizer.memory_efficient_batch_compress(
            data_list, max_memory_mb=10.0
        ))
        
        # Verify all items were processed
        self.assertEqual(len(results), len(data_list))
        
        # Verify result structure
        for idx, compressed_data, metrics in results:
            self.assertIsInstance(idx, int)
            self.assertIsNotNone(compressed_data)
            self.assertIsNotNone(metrics)
            self.assertGreater(metrics.compression_ratio, 0)
    
    def test_benchmark_compression_performance(self):
        """Test compression performance benchmarking."""
        # Define test sizes (small for quick testing)
        test_sizes = [
            ('small', (16, 16)),
            ('medium', (32, 32))
        ]
        
        # Run benchmark
        results = self.optimizer.benchmark_compression_performance(
            test_sizes, iterations=2
        )
        
        # Verify results structure
        self.assertIn('test_info', results)
        self.assertIn('results', results)
        self.assertEqual(results['test_info']['iterations'], 2)
        
        # Verify results for each test size
        for size_name, _ in test_sizes:
            self.assertIn(size_name, results['results'])
            size_results = results['results'][size_name]
            
            # Check required fields
            required_fields = [
                'shape', 'data_size_mb', 'compression_times', 
                'decompression_times', 'compression_ratios', 'stats'
            ]
            
            for field in required_fields:
                self.assertIn(field, size_results)
            
            # Check statistics
            stats = size_results['stats']
            self.assertGreater(stats['avg_compression_time'], 0)
            self.assertGreater(stats['avg_decompression_time'], 0)
            self.assertGreater(stats['avg_compression_ratio'], 0)
            self.assertGreater(stats['throughput_mb_per_sec'], 0)
    
    def test_optimize_pipeline_parameters(self):
        """Test pipeline parameter optimization."""
        # Create sample data
        sample_data = np.random.rand(32, 32).astype(np.float32)
        
        # Run optimization (with limited parameters for speed)
        results = self.optimizer.optimize_pipeline_parameters(
            sample_data, target_compression_ratio=2.0, target_quality_threshold=25.0
        )
        
        # Verify results structure
        self.assertIn('target_compression_ratio', results)
        self.assertIn('target_quality_threshold', results)
        self.assertIn('tested_configurations', results)
        self.assertIn('best_configuration', results)
        self.assertIn('best_score', results)
        
        # Verify configurations were tested
        self.assertGreater(len(results['tested_configurations']), 0)
        
        # Verify best configuration exists
        self.assertIsNotNone(results['best_configuration'])
        self.assertGreater(results['best_score'], 0)
        
        # Check best configuration structure
        best_config = results['best_configuration']
        required_fields = ['quality', 'block_size', 'normalization', 'metrics', 'combined_score']
        
        for field in required_fields:
            self.assertIn(field, best_config)
    
    def test_profile_pipeline_stages(self):
        """Test pipeline stage profiling."""
        # Create test data
        test_data = np.random.rand(24, 24).astype(np.float32)
        
        # Profile pipeline stages
        profiling_results = self.optimizer.profile_pipeline_stages(test_data)
        
        # Verify all stages are profiled
        expected_stages = [
            'preprocessing', 'dct_transform', 'quantization', 'huffman_encoding',
            'huffman_decoding', 'dequantization', 'inverse_dct', 'postprocessing'
        ]
        
        for stage in expected_stages:
            self.assertIn(stage, profiling_results)
            self.assertGreater(profiling_results[stage], 0)
        
        # Verify total time and percentages
        self.assertIn('total_time', profiling_results)
        self.assertGreater(profiling_results['total_time'], 0)
        
        # Check that percentages sum to approximately 100%
        total_percentage = sum(
            profiling_results[f'{stage}_percentage'] 
            for stage in expected_stages
        )
        self.assertAlmostEqual(total_percentage, 100.0, delta=1.0)
    
    def test_get_performance_recommendations(self):
        """Test performance recommendation generation."""
        # Create mock benchmark results
        mock_results = {
            'results': {
                'test_size': {
                    'data_size_mb': 1.0,
                    'stats': {
                        'throughput_mb_per_sec': 0.5,  # Low throughput
                        'avg_compression_ratio': 1.2,  # Low compression ratio
                        'avg_memory_usage': 5.0  # High memory usage relative to data size
                    }
                }
            }
        }
        
        # Get recommendations
        recommendations = self.optimizer.get_performance_recommendations(mock_results)
        
        # Verify recommendations were generated
        self.assertGreater(len(recommendations), 0)
        
        # Check that recommendations address the issues
        recommendation_text = ' '.join(recommendations).lower()
        self.assertIn('throughput', recommendation_text)
        self.assertIn('compression ratio', recommendation_text)
        self.assertIn('memory', recommendation_text)
    
    def test_chunk_compression_invalid_input(self):
        """Test error handling for invalid input in chunk compression."""
        # Test with 3D data (not supported)
        invalid_data = np.random.rand(10, 10, 3)
        
        with self.assertRaises(ValueError):
            list(self.optimizer.chunk_compress_large_data(invalid_data))
    
    def test_memory_efficient_processing_edge_cases(self):
        """Test memory-efficient processing with edge cases."""
        # Test with empty list
        results = list(self.optimizer.memory_efficient_batch_compress([], max_memory_mb=10.0))
        self.assertEqual(len(results), 0)
        
        # Test with very small memory limit
        data_list = [np.random.rand(4, 4).astype(np.float32)]
        results = list(self.optimizer.memory_efficient_batch_compress(data_list, max_memory_mb=0.001))
        self.assertEqual(len(results), 1)
    
    def test_benchmark_with_different_iterations(self):
        """Test benchmarking with different iteration counts."""
        test_sizes = [('tiny', (8, 8))]
        
        # Test with 1 iteration
        results_1 = self.optimizer.benchmark_compression_performance(test_sizes, iterations=1)
        self.assertEqual(len(results_1['results']['tiny']['compression_times']), 1)
        
        # Test with 3 iterations
        results_3 = self.optimizer.benchmark_compression_performance(test_sizes, iterations=3)
        self.assertEqual(len(results_3['results']['tiny']['compression_times']), 3)
    
    def test_optimization_with_different_targets(self):
        """Test parameter optimization with different target values."""
        sample_data = np.random.rand(16, 16).astype(np.float32)
        
        # Test with high compression ratio target
        results_high_ratio = self.optimizer.optimize_pipeline_parameters(
            sample_data, target_compression_ratio=5.0, target_quality_threshold=20.0
        )
        
        # Test with high quality target
        results_high_quality = self.optimizer.optimize_pipeline_parameters(
            sample_data, target_compression_ratio=1.5, target_quality_threshold=40.0
        )
        
        # Both should find valid configurations
        self.assertIsNotNone(results_high_ratio['best_configuration'])
        self.assertIsNotNone(results_high_quality['best_configuration'])
        
        # The configurations might be different due to different targets
        # (This is expected behavior)
    
    def test_profiling_consistency(self):
        """Test that profiling results are consistent across runs."""
        test_data = np.random.rand(16, 16).astype(np.float32)
        
        # Run profiling multiple times
        results_1 = self.optimizer.profile_pipeline_stages(test_data)
        results_2 = self.optimizer.profile_pipeline_stages(test_data)
        
        # Times should be similar (within reasonable variance)
        for stage in ['preprocessing', 'dct_transform', 'quantization']:
            ratio = results_1[stage] / results_2[stage]
            self.assertGreater(ratio, 0.1)  # Not more than 10x different
            self.assertLess(ratio, 10.0)    # Not more than 10x different


if __name__ == '__main__':
    unittest.main()