# Compression Pipeline

A comprehensive Python library for data compression using DCT (Discrete Cosine Transform), quantization, and Huffman encoding. This pipeline provides efficient lossy compression suitable for database storage, scientific data archival, and general-purpose data compression applications.

## Features

- **Multi-stage compression**: DCT transformation → Quantization → Huffman encoding
- **Configurable quality**: Adjustable compression vs quality tradeoffs
- **Multiple data types**: Support for numpy arrays, lists, text, binary data, and scalars
- **Comprehensive metrics**: Compression ratio, PSNR, SSIM, timing information
- **Performance optimization**: Memory-efficient processing, batch operations, chunked compression
- **Database integration**: Built-in SQLite storage with compression
- **Robust error handling**: Comprehensive validation and recovery mechanisms

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- scikit-image >= 0.18.0
- psutil >= 5.8.0

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package

```bash
# For development
pip install -e .

# Or copy the compression_pipeline directory to your project
```

## Quick Start

### Basic Usage

```python
import numpy as np
from compression_pipeline import CompressionPipeline

# Create sample data
data = np.random.rand(64, 64).astype(np.float32)

# Initialize compression pipeline
pipeline = CompressionPipeline(quality=0.8, block_size=8)

# Compress data
compressed_data = pipeline.compress(data)
print(f"Compression ratio: {data.nbytes / len(compressed_data.encoded_data):.2f}")

# Decompress data
reconstructed_data = pipeline.decompress(compressed_data)

# Calculate reconstruction error
mse = np.mean((data - reconstructed_data) ** 2)
print(f"Reconstruction MSE: {mse:.6f}")
```

### Compression with Metrics

```python
# Compress with comprehensive metrics
compressed_data, metrics = pipeline.compress_and_measure(data)

print(f"Compression Ratio: {metrics.compression_ratio:.2f}")
print(f"PSNR: {metrics.psnr:.2f} dB")
print(f"SSIM: {metrics.ssim:.4f}")
print(f"Compression Time: {metrics.compression_time:.4f} seconds")
```

### Different Data Types

```python
# The pipeline supports various data types
test_cases = {
    "numpy_array": np.random.rand(32, 32),
    "list_data": [[1, 2, 3], [4, 5, 6]],
    "text_data": "Hello, World! This is test data.",
    "binary_data": b"\\x00\\x01\\x02\\x03\\x04\\x05",
    "scalar": 42
}

for name, data in test_cases.items():
    compressed = pipeline.compress(data)
    reconstructed = pipeline.decompress(compressed)
    print(f"{name}: Compressed successfully")
```

## Configuration Options

### Pipeline Parameters

```python
pipeline = CompressionPipeline(
    block_size=8,           # DCT block size (4, 8, 16, 32)
    quality=0.8,            # Quality factor (0.1 to 1.0)
    normalization='minmax'  # Normalization method ('minmax', 'zscore', 'none')
)
```

### Quality Settings

- **0.1-0.3**: Maximum compression, suitable for previews
- **0.4-0.7**: Balanced compression and quality
- **0.8-0.95**: Minimal loss, best for archival

### Block Sizes

- **4x4**: Fast processing, good for small data
- **8x8**: Standard choice, good balance (default)
- **16x16+**: Better for large smooth regions

## Advanced Usage

### Performance Optimization

```python
from compression_pipeline.performance import PerformanceOptimizer

# Initialize optimizer
optimizer = PerformanceOptimizer(pipeline)

# Chunk-based compression for large data
large_data = np.random.rand(512, 512)
compressed_chunks = list(optimizer.chunk_compress_large_data(
    large_data, chunk_size=(64, 64), overlap=8
))

# Reconstruct from chunks
reconstructed = optimizer.reconstruct_from_chunks(
    compressed_chunks, large_data.shape, chunk_size=(64, 64), overlap=8
)
```

### Batch Processing

```python
# Memory-efficient batch compression
data_list = [np.random.rand(32, 32) for _ in range(10)]

for idx, compressed_data, metrics in optimizer.memory_efficient_batch_compress(
    data_list, max_memory_mb=100.0
):
    print(f"Item {idx}: {metrics.compression_ratio:.2f}x compression")
```

### Parameter Optimization

```python
# Automatic parameter optimization
sample_data = np.random.rand(64, 64)
optimization_results = optimizer.optimize_pipeline_parameters(
    sample_data,
    target_compression_ratio=2.0,
    target_quality_threshold=30.0
)

best_config = optimization_results['best_configuration']
print(f"Optimal quality: {best_config['quality']}")
print(f"Optimal block size: {best_config['block_size']}")
```

### Database Integration

```python
from examples.database_integration import CompressedDataStorage

# Initialize database storage
storage = CompressedDataStorage("my_data.db", compression_quality=0.7)

# Store compressed data
data = np.random.rand(100, 100)
storage_info = storage.store_data("my_dataset", data)
print(f"Space saved: {storage_info['space_saved_percent']:.1f}%")

# Retrieve data
retrieved_data = storage.retrieve_data("my_dataset")

# Get storage statistics
stats = storage.get_storage_statistics()
print(f"Total compression ratio: {stats['overall_compression_ratio']:.2f}")
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

- **basic_usage.py**: Fundamental operations and data type support
- **parameter_tuning.py**: Quality vs compression tradeoffs and optimization
- **database_integration.py**: SQLite integration with compressed storage
- **performance_benchmarks.py**: Comprehensive performance testing

Run examples:

```bash
python examples/basic_usage.py
python examples/parameter_tuning.py
python examples/database_integration.py
python examples/performance_benchmarks.py
```

## API Reference

### Core Classes

#### CompressionPipeline

Main pipeline class for compression operations.

```python
class CompressionPipeline:
    def __init__(self, block_size: int = 8, quality: float = 0.8,
                 normalization: str = 'minmax')
    def compress(self, data: Any) -> CompressedData
    def decompress(self, compressed_data: CompressedData) -> np.ndarray
    def compress_and_measure(self, data: Any) -> Tuple[CompressedData, CompressionMetrics]
    def get_metrics(self) -> Optional[CompressionMetrics]
    def batch_compress(self, data_list: list) -> Tuple[list, Dict[str, Any]]
```

#### CompressedData

Container for compressed data and metadata.

```python
@dataclass
class CompressedData:
    encoded_data: bytes
    huffman_table: Dict[int, str]
    original_shape: Tuple[int, ...]
    block_size: int
    quality: float
    quantization_table: np.ndarray
    metadata: Dict[str, Any]
```

#### CompressionMetrics

Comprehensive metrics for compression evaluation.

```python
@dataclass
class CompressionMetrics:
    compression_ratio: float
    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float
    mse: float  # Mean Squared Error
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
```

### Component Classes

- **DCTProcessor**: Handles DCT and inverse DCT operations
- **Quantizer**: Manages quantization and dequantization
- **HuffmanEncoder**: Implements Huffman encoding/decoding
- **DataPreprocessor**: Handles data validation and preprocessing
- **MetricsCollector**: Calculates performance and quality metrics
- **PerformanceOptimizer**: Provides optimization utilities

## Performance Characteristics

### Typical Performance (64x64 float32 data)

- **Compression Time**: ~0.01-0.05 seconds
- **Decompression Time**: ~0.005-0.02 seconds
- **Compression Ratio**: 2-8x (depending on data and quality)
- **Memory Usage**: ~2-3x input data size during processing

### Scalability

- **Time Complexity**: O(n) where n is data size
- **Memory Usage**: Linear with input size
- **Throughput**: 10-100 MB/s (depending on hardware and settings)

## Error Handling

The pipeline includes comprehensive error handling:

```python
from compression_pipeline.exceptions import (
    CompressionPipelineError, DataValidationError,
    CompressionError, DecompressionError
)

try:
    compressed_data = pipeline.compress(invalid_data)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
    # Get recovery suggestions
    suggestions = error_handler.get_error_recovery_suggestions(e)
    print("Suggestions:", suggestions['suggestions'])
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_integration.py

# Run with coverage
python -m pytest tests/ --cov=compression_pipeline
```

## Use Cases

### Scientific Data Archival

- Compress large datasets while preserving essential information
- Configurable quality for different precision requirements
- Batch processing for multiple experiments

### Database Storage Optimization

- Reduce storage requirements for large binary data
- Maintain data integrity with lossy compression
- Fast retrieval with efficient decompression

### IoT and Sensor Data

- Compress time-series data from sensors
- Balance compression ratio with reconstruction quality
- Memory-efficient processing for resource-constrained environments

### Image and Signal Processing

- Compress 2D data arrays (images, spectrograms)
- Preserve important frequency components
- Adjustable quality for different applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by JPEG compression algorithms
- Uses DCT implementation from SciPy
- SSIM calculation from scikit-image
- Performance monitoring with psutil

## Changelog

### Version 1.0.0

- Initial release
- Complete DCT-based compression pipeline
- Support for multiple data types
- Comprehensive metrics and optimization tools
- Database integration examples
- Full test coverage
