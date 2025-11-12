"""
Metrics collection system for the compression pipeline.

This module implements performance and quality measurements including compression ratio,
timing, MSE, PSNR, SSIM, and batch processing statistics.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from .models import CompressionMetrics
from skimage.metrics import structural_similarity as ssim


class MetricsCollector:
    """Collects and calculates performance and quality metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.batch_metrics: List[CompressionMetrics] = []
        self.current_compression_start: Optional[float] = None
        self.current_decompression_start: Optional[float] = None
    
    def start_compression_timer(self) -> None:
        """Start timing compression operation."""
        self.current_compression_start = time.perf_counter()
    
    def end_compression_timer(self) -> float:
        """
        End compression timing and return elapsed time.
        
        Returns:
            Compression time in seconds
            
        Raises:
            RuntimeError: If compression timer was not started
        """
        if self.current_compression_start is None:
            raise RuntimeError("Compression timer was not started")
        
        elapsed = time.perf_counter() - self.current_compression_start
        self.current_compression_start = None
        return elapsed
    
    def start_decompression_timer(self) -> None:
        """Start timing decompression operation."""
        self.current_decompression_start = time.perf_counter()
    
    def end_decompression_timer(self) -> float:
        """
        End decompression timing and return elapsed time.
        
        Returns:
            Decompression time in seconds
            
        Raises:
            RuntimeError: If decompression timer was not started
        """
        if self.current_decompression_start is None:
            raise RuntimeError("Decompression timer was not started")
        
        elapsed = time.perf_counter() - self.current_decompression_start
        self.current_decompression_start = None
        return elapsed
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Size of original data in bytes
            compressed_size: Size of compressed data in bytes
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    def calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error between original and reconstructed data.
        
        Args:
            original: Original data array
            reconstructed: Reconstructed data array
            
        Returns:
            Mean Squared Error
            
        Raises:
            ValueError: If arrays have different shapes
        """
        if original.shape != reconstructed.shape:
            raise ValueError("Original and reconstructed arrays must have the same shape")
        
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        return float(mse)
    
    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray, 
                      max_value: Optional[float] = None) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            original: Original data array
            reconstructed: Reconstructed data array
            max_value: Maximum possible value in the data (auto-detected if None)
            
        Returns:
            PSNR in decibels
        """
        mse = self.calculate_mse(original, reconstructed)
        
        if mse == 0:
            return float('inf')  # Perfect reconstruction
        
        if max_value is None:
            max_value = max(np.max(original), np.max(reconstructed))
        
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        return float(psnr)
    
    def calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index Measure.
        
        Args:
            original: Original data array
            reconstructed: Reconstructed data array
            
        Returns:
            SSIM value between -1 and 1 (1 = perfect similarity)
            
        Raises:
            ValueError: If arrays are not suitable for SSIM calculation
        """
        if original.shape != reconstructed.shape:
            raise ValueError("Original and reconstructed arrays must have the same shape")
        
        # Convert to float for SSIM calculation
        orig_float = original.astype(np.float64)
        recon_float = reconstructed.astype(np.float64)
        
        # Handle different array dimensions
        if orig_float.ndim == 1:
            # For 1D arrays, reshape to 2D for SSIM
            size = int(np.sqrt(len(orig_float)))
            if size * size == len(orig_float):
                orig_2d = orig_float.reshape(size, size)
                recon_2d = recon_float.reshape(size, size)
            else:
                # Pad to make square
                pad_size = int(np.ceil(np.sqrt(len(orig_float))))
                padded_orig = np.pad(orig_float, (0, pad_size**2 - len(orig_float)), mode='edge')
                padded_recon = np.pad(recon_float, (0, pad_size**2 - len(recon_float)), mode='edge')
                orig_2d = padded_orig.reshape(pad_size, pad_size)
                recon_2d = padded_recon.reshape(pad_size, pad_size)
        
        elif orig_float.ndim == 2:
            orig_2d = orig_float
            recon_2d = recon_float
        
        else:
            raise ValueError("SSIM calculation supports only 1D and 2D arrays")
        
        # Calculate data range for SSIM
        data_range = max(np.max(orig_2d) - np.min(orig_2d), 
                        np.max(recon_2d) - np.min(recon_2d))
        
        if data_range == 0:
            return 1.0  # Identical constant arrays
        
        try:
            ssim_value = ssim(orig_2d, recon_2d, data_range=data_range)
            return float(ssim_value)
        except Exception as e:
            # Fallback to simplified SSIM calculation
            return self._simple_ssim(orig_2d, recon_2d)
    
    def _simple_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Simplified SSIM calculation as fallback.
        
        Args:
            original: Original 2D array
            reconstructed: Reconstructed 2D array
            
        Returns:
            Simplified SSIM value
        """
        # Calculate means
        mu1 = np.mean(original)
        mu2 = np.mean(reconstructed)
        
        # Calculate variances and covariance
        var1 = np.var(original)
        var2 = np.var(reconstructed)
        cov = np.mean((original - mu1) * (reconstructed - mu2))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        
        if denominator == 0:
            return 1.0
        
        return numerator / denominator
    
    def create_metrics(self, original_data: np.ndarray, reconstructed_data: np.ndarray,
                      compressed_size: int, compression_time: float, 
                      decompression_time: float) -> CompressionMetrics:
        """
        Create comprehensive metrics for a compression/decompression cycle.
        
        Args:
            original_data: Original input data
            reconstructed_data: Reconstructed data after compression/decompression
            compressed_size: Size of compressed data in bytes
            compression_time: Time taken for compression
            decompression_time: Time taken for decompression
            
        Returns:
            CompressionMetrics object with all calculated metrics
        """
        original_size = original_data.nbytes
        compression_ratio = self.calculate_compression_ratio(original_size, compressed_size)
        mse = self.calculate_mse(original_data, reconstructed_data)
        psnr = self.calculate_psnr(original_data, reconstructed_data)
        ssim_value = self.calculate_ssim(original_data, reconstructed_data)
        
        metrics = CompressionMetrics(
            compression_ratio=compression_ratio,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            decompression_time=decompression_time,
            mse=mse,
            psnr=psnr,
            ssim=ssim_value
        )
        
        return metrics
    
    def add_batch_metrics(self, metrics: CompressionMetrics) -> None:
        """
        Add metrics to batch collection for aggregate statistics.
        
        Args:
            metrics: CompressionMetrics to add to batch
        """
        self.batch_metrics.append(metrics)
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Calculate aggregate statistics for batch processing.
        
        Returns:
            Dictionary with batch statistics
        """
        if not self.batch_metrics:
            return {}
        
        # Extract values for each metric
        compression_ratios = [m.compression_ratio for m in self.batch_metrics]
        compression_times = [m.compression_time for m in self.batch_metrics]
        decompression_times = [m.decompression_time for m in self.batch_metrics]
        mse_values = [m.mse for m in self.batch_metrics]
        psnr_values = [m.psnr for m in self.batch_metrics if not np.isinf(m.psnr)]
        ssim_values = [m.ssim for m in self.batch_metrics]
        
        # Calculate statistics
        stats = {
            'batch_size': len(self.batch_metrics),
            'compression_ratio': {
                'mean': np.mean(compression_ratios),
                'std': np.std(compression_ratios),
                'min': np.min(compression_ratios),
                'max': np.max(compression_ratios),
                'median': np.median(compression_ratios)
            },
            'compression_time': {
                'mean': np.mean(compression_times),
                'std': np.std(compression_times),
                'min': np.min(compression_times),
                'max': np.max(compression_times),
                'total': np.sum(compression_times)
            },
            'decompression_time': {
                'mean': np.mean(decompression_times),
                'std': np.std(decompression_times),
                'min': np.min(decompression_times),
                'max': np.max(decompression_times),
                'total': np.sum(decompression_times)
            },
            'mse': {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            },
            'ssim': {
                'mean': np.mean(ssim_values),
                'std': np.std(ssim_values),
                'min': np.min(ssim_values),
                'max': np.max(ssim_values)
            }
        }
        
        # Add PSNR stats if we have valid values
        if psnr_values:
            stats['psnr'] = {
                'mean': np.mean(psnr_values),
                'std': np.std(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values)
            }
        
        return stats
    
    def clear_batch_metrics(self) -> None:
        """Clear collected batch metrics."""
        self.batch_metrics.clear()
    
    def export_metrics_to_dict(self, metrics: CompressionMetrics) -> Dict[str, Any]:
        """
        Export metrics to dictionary format.
        
        Args:
            metrics: CompressionMetrics to export
            
        Returns:
            Dictionary representation of metrics
        """
        return {
            'compression_ratio': metrics.compression_ratio,
            'original_size_bytes': metrics.original_size,
            'compressed_size_bytes': metrics.compressed_size,
            'compression_time_seconds': metrics.compression_time,
            'decompression_time_seconds': metrics.decompression_time,
            'mse': metrics.mse,
            'psnr_db': metrics.psnr,
            'ssim': metrics.ssim,
            'space_savings_percent': (1 - 1/metrics.compression_ratio) * 100 if metrics.compression_ratio > 0 else 0
        }
    
    def compare_metrics(self, metrics1: CompressionMetrics, 
                       metrics2: CompressionMetrics) -> Dict[str, float]:
        """
        Compare two sets of metrics.
        
        Args:
            metrics1: First metrics set
            metrics2: Second metrics set
            
        Returns:
            Dictionary with comparison results (positive = metrics1 is better)
        """
        return {
            'compression_ratio_diff': metrics1.compression_ratio - metrics2.compression_ratio,
            'compression_time_diff': metrics2.compression_time - metrics1.compression_time,  # Lower is better
            'decompression_time_diff': metrics2.decompression_time - metrics1.decompression_time,  # Lower is better
            'mse_diff': metrics2.mse - metrics1.mse,  # Lower is better
            'psnr_diff': metrics1.psnr - metrics2.psnr,
            'ssim_diff': metrics1.ssim - metrics2.ssim
        }