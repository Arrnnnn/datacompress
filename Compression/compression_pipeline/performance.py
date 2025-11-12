"""
Performance optimization utilities for the compression pipeline.

This module provides memory-efficient processing, performance benchmarks,
and optimization strategies for large datasets.
"""

import time
import numpy as np
import gc
from typing import Iterator, List, Dict, Any, Tuple, Optional, Union
from .pipeline import CompressionPipeline
from .models import CompressionMetrics


class PerformanceOptimizer:
    """Performance optimization utilities for compression pipeline."""
    
    def __init__(self, pipeline: CompressionPipeline):
        """
        Initialize performance optimizer.
        
        Args:
            pipeline: CompressionPipeline instance to optimize
        """
        self.pipeline = pipeline
        self.benchmark_results: List[Dict[str, Any]] = []
    
    def chunk_compress_large_data(self, data: np.ndarray, 
                                 chunk_size: Tuple[int, int] = (64, 64),
                                 overlap: int = 8) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Compress large data in chunks for memory efficiency.
        
        Args:
            data: Large input data array
            chunk_size: Size of each chunk (height, width)
            overlap: Overlap between chunks to avoid boundary artifacts
            
        Yields:
            Tuples of (compressed_chunk, position)
        """
        if data.ndim != 2:
            raise ValueError("Chunk compression currently supports only 2D arrays")
        
        height, width = data.shape
        chunk_h, chunk_w = chunk_size
        
        # Calculate chunk positions with overlap
        for i in range(0, height, chunk_h - overlap):
            for j in range(0, width, chunk_w - overlap):
                # Calculate actual chunk boundaries
                end_i = min(i + chunk_h, height)
                end_j = min(j + chunk_w, width)
                
                # Extract chunk
                chunk = data[i:end_i, j:end_j]
                
                # Compress chunk
                compressed_chunk = self.pipeline.compress(chunk)
                
                yield compressed_chunk, (i, j)
                
                # Force garbage collection to manage memory
                gc.collect()
    
    def reconstruct_from_chunks(self, compressed_chunks: List[Tuple[np.ndarray, Tuple[int, int]]],
                               original_shape: Tuple[int, int],
                               chunk_size: Tuple[int, int] = (64, 64),
                               overlap: int = 8) -> np.ndarray:
        """
        Reconstruct large data from compressed chunks.
        
        Args:
            compressed_chunks: List of (compressed_chunk, position) tuples
            original_shape: Original data shape
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            Reconstructed data array
        """
        height, width = original_shape
        reconstructed = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        chunk_h, chunk_w = chunk_size
        
        for compressed_chunk, (i, j) in compressed_chunks:
            # Decompress chunk
            decompressed_chunk = self.pipeline.decompress(compressed_chunk)
            
            # Calculate boundaries
            end_i = min(i + decompressed_chunk.shape[0], height)
            end_j = min(j + decompressed_chunk.shape[1], width)
            
            # Create weight matrix for blending overlaps
            chunk_height = end_i - i
            chunk_width = end_j - j
            weights = np.ones((chunk_height, chunk_width))
            
            # Apply tapering at edges for smooth blending
            if overlap > 0:
                # Taper top edge
                if i > 0:
                    taper_size = min(overlap, chunk_height)
                    for k in range(taper_size):
                        weights[k, :] *= (k + 1) / taper_size
                
                # Taper left edge
                if j > 0:
                    taper_size = min(overlap, chunk_width)
                    for k in range(taper_size):
                        weights[:, k] *= (k + 1) / taper_size
            
            # Add to reconstruction with weights
            reconstructed[i:end_i, j:end_j] += decompressed_chunk[:chunk_height, :chunk_width] * weights
            weight_map[i:end_i, j:end_j] += weights
        
        # Normalize by weights to handle overlaps
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        reconstructed /= weight_map
        
        return reconstructed
    
    def memory_efficient_batch_compress(self, data_list: List[np.ndarray],
                                       max_memory_mb: float = 500.0) -> Iterator[Tuple[int, Any, CompressionMetrics]]:
        """
        Memory-efficient batch compression with automatic memory management.
        
        Args:
            data_list: List of data arrays to compress
            max_memory_mb: Maximum memory usage in MB
            
        Yields:
            Tuples of (index, compressed_data, metrics)
        """
        current_memory_mb = 0.0
        batch_buffer = []
        
        for idx, data in enumerate(data_list):
            # Estimate memory usage for this item
            if isinstance(data, np.ndarray):
                item_memory_mb = data.nbytes / (1024 * 1024)
            else:
                # Rough estimate for other types
                item_memory_mb = len(str(data)) / (1024 * 1024)
            
            # Check if adding this item would exceed memory limit
            if current_memory_mb + item_memory_mb > max_memory_mb and batch_buffer:
                # Process current batch
                yield from self._process_memory_batch(batch_buffer)
                
                # Clear batch and reset memory counter
                batch_buffer.clear()
                current_memory_mb = 0.0
                gc.collect()
            
            # Add item to batch
            batch_buffer.append((idx, data))
            current_memory_mb += item_memory_mb
        
        # Process remaining items
        if batch_buffer:
            yield from self._process_memory_batch(batch_buffer)
    
    def _process_memory_batch(self, batch_buffer: List[Tuple[int, np.ndarray]]) -> Iterator[Tuple[int, Any, CompressionMetrics]]:
        """Process a batch of data items."""
        for idx, data in batch_buffer:
            compressed_data, metrics = self.pipeline.compress_and_measure(data)
            yield idx, compressed_data, metrics
    
    def benchmark_compression_performance(self, test_data_sizes: List[Tuple[str, Tuple[int, ...]]],
                                        iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark compression performance across different data sizes.
        
        Args:
            test_data_sizes: List of (name, shape) tuples for test data
            iterations: Number of iterations per test
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            'test_info': {
                'iterations': iterations,
                'pipeline_config': self.pipeline.get_pipeline_info()
            },
            'results': {}
        }
        
        for size_name, shape in test_data_sizes:
            print(f"Benchmarking {size_name} ({shape})...")
            
            size_results = {
                'shape': shape,
                'data_size_mb': 0,
                'compression_times': [],
                'decompression_times': [],
                'compression_ratios': [],
                'memory_usage': []
            }
            
            for iteration in range(iterations):
                # Generate test data
                test_data = np.random.rand(*shape).astype(np.float32)
                size_results['data_size_mb'] = test_data.nbytes / (1024 * 1024)
                
                # Measure memory before compression
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024 * 1024)
                
                # Benchmark compression
                start_time = time.perf_counter()
                compressed_data = self.pipeline.compress(test_data)
                compression_time = time.perf_counter() - start_time
                
                # Measure memory after compression
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_usage = memory_after - memory_before
                
                # Benchmark decompression
                start_time = time.perf_counter()
                reconstructed_data = self.pipeline.decompress(compressed_data)
                decompression_time = time.perf_counter() - start_time
                
                # Calculate compression ratio
                compression_ratio = test_data.nbytes / len(compressed_data.encoded_data)
                
                # Store results
                size_results['compression_times'].append(compression_time)
                size_results['decompression_times'].append(decompression_time)
                size_results['compression_ratios'].append(compression_ratio)
                size_results['memory_usage'].append(memory_usage)
                
                # Clean up
                del test_data, compressed_data, reconstructed_data
                gc.collect()
            
            # Calculate statistics
            size_results['stats'] = {
                'avg_compression_time': np.mean(size_results['compression_times']),
                'std_compression_time': np.std(size_results['compression_times']),
                'avg_decompression_time': np.mean(size_results['decompression_times']),
                'std_decompression_time': np.std(size_results['decompression_times']),
                'avg_compression_ratio': np.mean(size_results['compression_ratios']),
                'std_compression_ratio': np.std(size_results['compression_ratios']),
                'avg_memory_usage': np.mean(size_results['memory_usage']),
                'throughput_mb_per_sec': size_results['data_size_mb'] / np.mean(size_results['compression_times'])
            }
            
            benchmark_results['results'][size_name] = size_results
        
        self.benchmark_results.append(benchmark_results)
        return benchmark_results
    
    def optimize_pipeline_parameters(self, sample_data: np.ndarray,
                                   target_compression_ratio: float = 2.0,
                                   target_quality_threshold: float = 30.0) -> Dict[str, Any]:
        """
        Optimize pipeline parameters for given constraints.
        
        Args:
            sample_data: Sample data for optimization
            target_compression_ratio: Target compression ratio
            target_quality_threshold: Minimum PSNR threshold
            
        Returns:
            Dictionary with optimized parameters and results
        """
        optimization_results = {
            'target_compression_ratio': target_compression_ratio,
            'target_quality_threshold': target_quality_threshold,
            'tested_configurations': [],
            'best_configuration': None,
            'best_score': -1
        }
        
        # Test different parameter combinations
        quality_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
        block_sizes = [4, 8, 16]
        normalization_methods = ['minmax', 'zscore', 'none']
        
        for quality in quality_levels:
            for block_size in block_sizes:
                for normalization in normalization_methods:
                    # Create pipeline with test parameters
                    test_pipeline = CompressionPipeline(
                        quality=quality,
                        block_size=block_size,
                        normalization=normalization
                    )
                    
                    try:
                        # Test compression
                        compressed_data, metrics = test_pipeline.compress_and_measure(sample_data)
                        
                        # Calculate score based on constraints
                        compression_ratio_score = min(metrics.compression_ratio / target_compression_ratio, 1.0)
                        quality_score = min(metrics.psnr / target_quality_threshold, 1.0) if metrics.psnr < float('inf') else 1.0
                        
                        # Combined score (weighted average)
                        combined_score = 0.6 * compression_ratio_score + 0.4 * quality_score
                        
                        config_result = {
                            'quality': quality,
                            'block_size': block_size,
                            'normalization': normalization,
                            'metrics': metrics,
                            'compression_ratio_score': compression_ratio_score,
                            'quality_score': quality_score,
                            'combined_score': combined_score
                        }
                        
                        optimization_results['tested_configurations'].append(config_result)
                        
                        # Update best configuration
                        if combined_score > optimization_results['best_score']:
                            optimization_results['best_score'] = combined_score
                            optimization_results['best_configuration'] = config_result
                    
                    except Exception as e:
                        # Log failed configuration
                        optimization_results['tested_configurations'].append({
                            'quality': quality,
                            'block_size': block_size,
                            'normalization': normalization,
                            'error': str(e)
                        })
        
        return optimization_results
    
    def profile_pipeline_stages(self, test_data: np.ndarray) -> Dict[str, float]:
        """
        Profile individual pipeline stages to identify bottlenecks.
        
        Args:
            test_data: Test data for profiling
            
        Returns:
            Dictionary with timing for each stage
        """
        profiling_results = {}
        
        # Profile data preprocessing
        start_time = time.perf_counter()
        validated_data = self.pipeline.data_preprocessor.validate_and_convert(test_data)
        normalized_data = self.pipeline.data_preprocessor.normalize_data(validated_data, self.pipeline.normalization)
        padded_data, original_shape = self.pipeline.data_preprocessor.pad_to_block_size(normalized_data, self.pipeline.block_size)
        profiling_results['preprocessing'] = time.perf_counter() - start_time
        
        # Profile DCT transformation
        start_time = time.perf_counter()
        dct_coefficients = self.pipeline.dct_processor.process_blocks(padded_data, inverse=False)
        profiling_results['dct_transform'] = time.perf_counter() - start_time
        
        # Profile quantization
        start_time = time.perf_counter()
        quantized_coeffs = self.pipeline._quantize_blocks(dct_coefficients)
        profiling_results['quantization'] = time.perf_counter() - start_time
        
        # Profile Huffman encoding
        start_time = time.perf_counter()
        encoded_data, huffman_table = self.pipeline.huffman_encoder.encode(quantized_coeffs)
        profiling_results['huffman_encoding'] = time.perf_counter() - start_time
        
        # Profile decompression stages
        start_time = time.perf_counter()
        decoded_coeffs = self.pipeline.huffman_encoder.decode(encoded_data, huffman_table, quantized_coeffs.shape)
        profiling_results['huffman_decoding'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        dequantized_coeffs = self.pipeline._dequantize_blocks(decoded_coeffs)
        profiling_results['dequantization'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        reconstructed_padded = self.pipeline.dct_processor.process_blocks(dequantized_coeffs, inverse=True)
        profiling_results['inverse_dct'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        reconstructed_data = self.pipeline.data_preprocessor.remove_padding(reconstructed_padded, original_shape)
        profiling_results['postprocessing'] = time.perf_counter() - start_time
        
        # Calculate total and percentages
        total_time = sum(profiling_results.values())
        profiling_results['total_time'] = total_time
        
        for stage, stage_time in profiling_results.items():
            if stage != 'total_time':
                profiling_results[f'{stage}_percentage'] = (stage_time / total_time) * 100
        
        return profiling_results
    
    def get_performance_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """
        Generate performance optimization recommendations based on benchmark results.
        
        Args:
            benchmark_results: Results from benchmark_compression_performance
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze results for patterns
        for size_name, results in benchmark_results['results'].items():
            stats = results['stats']
            
            # Check compression speed
            if stats['throughput_mb_per_sec'] < 1.0:
                recommendations.append(
                    f"Low throughput ({stats['throughput_mb_per_sec']:.2f} MB/s) for {size_name}. "
                    "Consider using smaller block sizes or reducing quality for faster compression."
                )
            
            # Check compression ratio
            if stats['avg_compression_ratio'] < 1.5:
                recommendations.append(
                    f"Low compression ratio ({stats['avg_compression_ratio']:.2f}) for {size_name}. "
                    "Consider using lower quality settings or different normalization method."
                )
            
            # Check memory usage
            if stats['avg_memory_usage'] > results['data_size_mb'] * 3:
                recommendations.append(
                    f"High memory usage ({stats['avg_memory_usage']:.1f} MB) for {size_name}. "
                    "Consider using chunk-based processing for large datasets."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance looks good! No specific optimizations needed.")
        
        return recommendations