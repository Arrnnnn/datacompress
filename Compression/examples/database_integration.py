"""
Database integration example for the compression pipeline.

This script demonstrates how to integrate the compression pipeline with
database systems for efficient storage of large datasets.
"""

import sqlite3
import numpy as np
import pickle
import json
import time
from typing import List, Dict, Any, Optional
from compression_pipeline import CompressionPipeline
from compression_pipeline.models import CompressedData


class CompressedDataStorage:
    """Database storage system with integrated compression."""
    
    def __init__(self, db_path: str = "compressed_data.db", 
                 compression_quality: float = 0.7):
        """
        Initialize compressed data storage.
        
        Args:
            db_path: Path to SQLite database file
            compression_quality: Quality setting for compression
        """
        self.db_path = db_path
        self.pipeline = CompressionPipeline(quality=compression_quality)
        self.conn = None
        self._setup_database()
    
    def _setup_database(self):
        """Set up database tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create table for compressed data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compressed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                original_shape TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                quality_metrics TEXT,
                compressed_blob BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_name ON compressed_data(name)
        ''')
        
        self.conn.commit()
    
    def store_data(self, name: str, data: np.ndarray, 
                   calculate_metrics: bool = True) -> Dict[str, Any]:
        """
        Store data with compression in the database.
        
        Args:
            name: Unique identifier for the data
            data: Data to store
            calculate_metrics: Whether to calculate quality metrics
            
        Returns:
            Dictionary with storage information
        """
        print(f"Storing data '{name}'...")
        
        # Compress data
        if calculate_metrics:
            compressed_data, metrics = self.pipeline.compress_and_measure(data)
            quality_metrics = {
                'mse': metrics.mse,
                'psnr': metrics.psnr,
                'ssim': metrics.ssim,
                'compression_time': metrics.compression_time,
                'decompression_time': metrics.decompression_time
            }
        else:
            compressed_data = self.pipeline.compress(data)
            quality_metrics = {}
        
        # Serialize compressed data
        compressed_blob = pickle.dumps(compressed_data)
        
        # Prepare metadata
        metadata = {
            'block_size': compressed_data.block_size,
            'quality': compressed_data.quality,
            'normalization_method': compressed_data.metadata.get('normalization_method'),
            'original_dtype': compressed_data.metadata.get('original_dtype')
        }
        
        # Store in database
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO compressed_data 
                (name, original_shape, original_size, compressed_size, 
                 compression_ratio, quality_metrics, compressed_blob, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name,
                str(data.shape),
                data.nbytes,
                len(compressed_blob),
                data.nbytes / len(compressed_blob),
                json.dumps(quality_metrics),
                compressed_blob,
                json.dumps(metadata)
            ))
            
            self.conn.commit()
            
            storage_info = {
                'name': name,
                'original_size': data.nbytes,
                'compressed_size': len(compressed_blob),
                'compression_ratio': data.nbytes / len(compressed_blob),
                'space_saved_bytes': data.nbytes - len(compressed_blob),
                'space_saved_percent': (1 - len(compressed_blob) / data.nbytes) * 100
            }
            
            print(f"  Stored successfully: {storage_info['compression_ratio']:.2f}x compression, "
                  f"{storage_info['space_saved_percent']:.1f}% space saved")
            
            return storage_info
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
    
    def retrieve_data(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve and decompress data from the database.
        
        Args:
            name: Identifier of the data to retrieve
            
        Returns:
            Decompressed data or None if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT compressed_blob FROM compressed_data WHERE name = ?
        ''', (name,))
        
        result = cursor.fetchone()
        
        if result is None:
            print(f"Data '{name}' not found in database")
            return None
        
        # Deserialize and decompress
        compressed_blob = result[0]
        compressed_data = pickle.loads(compressed_blob)
        
        reconstructed_data = self.pipeline.decompress(compressed_data)
        
        print(f"Retrieved data '{name}' successfully")
        return reconstructed_data
    
    def list_stored_data(self) -> List[Dict[str, Any]]:
        """
        List all stored data with statistics.
        
        Returns:
            List of dictionaries with data information
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT name, original_shape, original_size, compressed_size, 
                   compression_ratio, quality_metrics, created_at
            FROM compressed_data
            ORDER BY created_at DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            quality_metrics = json.loads(row[5]) if row[5] else {}
            
            results.append({
                'name': row[0],
                'original_shape': row[1],
                'original_size': row[2],
                'compressed_size': row[3],
                'compression_ratio': row[4],
                'quality_metrics': quality_metrics,
                'created_at': row[6]
            })
        
        return results
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get overall storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_items,
                SUM(original_size) as total_original_size,
                SUM(compressed_size) as total_compressed_size,
                AVG(compression_ratio) as avg_compression_ratio,
                MIN(compression_ratio) as min_compression_ratio,
                MAX(compression_ratio) as max_compression_ratio
            FROM compressed_data
        ''')
        
        result = cursor.fetchone()
        
        if result[0] == 0:
            return {'total_items': 0}
        
        total_original = result[1]
        total_compressed = result[2]
        
        return {
            'total_items': result[0],
            'total_original_size_mb': total_original / (1024 * 1024),
            'total_compressed_size_mb': total_compressed / (1024 * 1024),
            'total_space_saved_mb': (total_original - total_compressed) / (1024 * 1024),
            'overall_compression_ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'avg_compression_ratio': result[3],
            'min_compression_ratio': result[4],
            'max_compression_ratio': result[5],
            'space_saved_percent': (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        }
    
    def delete_data(self, name: str) -> bool:
        """
        Delete data from the database.
        
        Args:
            name: Identifier of the data to delete
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute('DELETE FROM compressed_data WHERE name = ?', (name,))
        
        if cursor.rowcount > 0:
            self.conn.commit()
            print(f"Deleted data '{name}'")
            return True
        else:
            print(f"Data '{name}' not found")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def demonstrate_basic_storage():
    """Demonstrate basic storage and retrieval operations."""
    print("=== Basic Storage Operations ===")
    
    # Initialize storage system
    storage = CompressedDataStorage("example_database.db", compression_quality=0.7)
    
    # Create sample datasets
    datasets = {
        "random_data": np.random.rand(64, 64).astype(np.float32),
        "structured_data": np.outer(np.sin(np.linspace(0, 4*np.pi, 32)), 
                                   np.cos(np.linspace(0, 4*np.pi, 32))).astype(np.float32),
        "sparse_data": np.zeros((48, 48), dtype=np.float32),
        "noisy_signal": np.sin(np.linspace(0, 10*np.pi, 100)) + 0.1 * np.random.randn(100)
    }
    
    # Add some structure to sparse data
    datasets["sparse_data"][10:20, 10:20] = 1.0
    datasets["sparse_data"][30:40, 30:40] = 0.5
    
    # Store all datasets
    storage_results = {}
    for name, data in datasets.items():
        storage_info = storage.store_data(name, data, calculate_metrics=True)
        storage_results[name] = storage_info
    
    # Retrieve and verify data
    print("\\nVerifying stored data...")
    for name in datasets.keys():
        original_data = datasets[name]
        retrieved_data = storage.retrieve_data(name)
        
        if retrieved_data is not None:
            mse = np.mean((original_data - retrieved_data) ** 2)
            print(f"  {name}: MSE = {mse:.6f}")
        else:
            print(f"  {name}: Retrieval failed")
    
    storage.close()
    return storage_results


def demonstrate_batch_operations():
    """Demonstrate batch storage operations."""
    print("\\n=== Batch Operations ===")
    
    storage = CompressedDataStorage("batch_example.db", compression_quality=0.6)
    
    # Generate batch of data
    print("Generating batch data...")
    batch_data = {}
    for i in range(10):
        # Create different types of data
        if i % 3 == 0:
            data = np.random.rand(32, 32).astype(np.float32)
            data_type = "random"
        elif i % 3 == 1:
            x = np.linspace(0, 2*np.pi, 32)
            data = np.outer(np.sin(x * (i+1)), np.cos(x * (i+1))).astype(np.float32)
            data_type = "sinusoidal"
        else:
            data = np.ones((32, 32), dtype=np.float32) * (i + 1) / 10
            data_type = "constant"
        
        batch_data[f"{data_type}_{i:02d}"] = data
    
    # Store batch with timing
    print(f"\\nStoring {len(batch_data)} datasets...")
    start_time = time.time()
    
    for name, data in batch_data.items():
        storage.store_data(name, data, calculate_metrics=False)  # Skip metrics for speed
    
    storage_time = time.time() - start_time
    print(f"Batch storage completed in {storage_time:.2f} seconds")
    
    # Get statistics
    stats = storage.get_storage_statistics()
    print(f"\\nStorage Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Total original size: {stats['total_original_size_mb']:.2f} MB")
    print(f"  Total compressed size: {stats['total_compressed_size_mb']:.2f} MB")
    print(f"  Space saved: {stats['total_space_saved_mb']:.2f} MB ({stats['space_saved_percent']:.1f}%)")
    print(f"  Overall compression ratio: {stats['overall_compression_ratio']:.2f}")
    
    storage.close()
    return stats


def demonstrate_performance_comparison():
    """Compare storage performance with and without compression."""
    print("\\n=== Performance Comparison ===")
    
    # Create test data
    test_data = np.random.rand(100, 100).astype(np.float32)
    
    # Test uncompressed storage (using pickle directly)
    print("Testing uncompressed storage...")
    start_time = time.time()
    uncompressed_blob = pickle.dumps(test_data)
    uncompressed_time = time.time() - start_time
    uncompressed_size = len(uncompressed_blob)
    
    # Test compressed storage
    print("Testing compressed storage...")
    pipeline = CompressionPipeline(quality=0.7)
    
    start_time = time.time()
    compressed_data = pipeline.compress(test_data)
    compressed_blob = pickle.dumps(compressed_data)
    compression_time = time.time() - start_time
    compressed_size = len(compressed_blob)
    
    # Test decompression
    start_time = time.time()
    loaded_compressed = pickle.loads(compressed_blob)
    reconstructed_data = pipeline.decompress(loaded_compressed)
    decompression_time = time.time() - start_time
    
    # Calculate metrics
    mse = np.mean((test_data - reconstructed_data) ** 2)
    compression_ratio = uncompressed_size / compressed_size
    
    print(f"\\nPerformance Comparison Results:")
    print(f"  Uncompressed size: {uncompressed_size} bytes")
    print(f"  Compressed size: {compressed_size} bytes")
    print(f"  Compression ratio: {compression_ratio:.2f}")
    print(f"  Space saved: {(1 - 1/compression_ratio)*100:.1f}%")
    print(f"  Compression time: {compression_time:.4f} seconds")
    print(f"  Decompression time: {decompression_time:.4f} seconds")
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    return {
        'uncompressed_size': uncompressed_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'compression_time': compression_time,
        'decompression_time': decompression_time,
        'mse': mse
    }


def demonstrate_data_lifecycle():
    """Demonstrate complete data lifecycle in database."""
    print("\\n=== Data Lifecycle Management ===")
    
    storage = CompressedDataStorage("lifecycle_example.db", compression_quality=0.8)
    
    # Create and store initial data
    data_name = "lifecycle_test_data"
    original_data = np.random.rand(50, 50).astype(np.float32)
    
    print(f"1. Storing initial data...")
    storage.store_data(data_name, original_data)
    
    # List stored data
    print(f"\\n2. Listing stored data...")
    stored_items = storage.list_stored_data()
    for item in stored_items:
        print(f"   {item['name']}: {item['original_shape']}, "
              f"ratio={item['compression_ratio']:.2f}")
    
    # Retrieve and modify data
    print(f"\\n3. Retrieving and modifying data...")
    retrieved_data = storage.retrieve_data(data_name)
    
    if retrieved_data is not None:
        # Modify data (add some structure)
        modified_data = retrieved_data + 0.5 * np.sin(
            np.outer(np.linspace(0, 4*np.pi, 50), np.linspace(0, 4*np.pi, 50))
        )
        
        # Store modified version
        modified_name = f"{data_name}_modified"
        storage.store_data(modified_name, modified_data)
    
    # Get final statistics
    print(f"\\n4. Final storage statistics...")
    final_stats = storage.get_storage_statistics()
    print(f"   Total items: {final_stats['total_items']}")
    print(f"   Total space saved: {final_stats['total_space_saved_mb']:.2f} MB")
    
    # Cleanup
    print(f"\\n5. Cleaning up...")
    storage.delete_data(data_name)
    storage.delete_data(modified_name)
    
    storage.close()


def main():
    """Run all database integration examples."""
    print("Compression Pipeline - Database Integration Examples")
    print("=" * 60)
    
    # Basic storage operations
    storage_results = demonstrate_basic_storage()
    
    # Batch operations
    batch_stats = demonstrate_batch_operations()
    
    # Performance comparison
    perf_results = demonstrate_performance_comparison()
    
    # Data lifecycle
    demonstrate_data_lifecycle()
    
    # Summary
    print("\\n=== Integration Summary ===")
    print("Successfully demonstrated:")
    print("- Compressed data storage in SQLite database")
    print("- Batch operations with timing")
    print("- Performance comparison vs uncompressed storage")
    print("- Complete data lifecycle management")
    
    print(f"\\nKey Benefits:")
    print(f"- Average compression ratio: {perf_results['compression_ratio']:.2f}x")
    print(f"- Space savings: {(1 - 1/perf_results['compression_ratio'])*100:.1f}%")
    print(f"- Fast compression: {perf_results['compression_time']:.4f}s")
    print(f"- Fast decompression: {perf_results['decompression_time']:.4f}s")
    
    print("\\nUse Cases:")
    print("- Scientific data archival")
    print("- Image/signal processing pipelines")
    print("- IoT sensor data storage")
    print("- Machine learning dataset compression")


if __name__ == "__main__":
    main()