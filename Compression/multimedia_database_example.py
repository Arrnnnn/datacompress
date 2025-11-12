#!/usr/bin/env python3
"""
Multi-Media Database Compression Example

This script demonstrates how to compress a database containing multiple data types:
- Images (JPEG, PNG, etc.)
- Videos (MP4, AVI, etc.)
- Text documents
- Binary files
- Numerical datasets

Usage:
    python multimedia_database_example.py
"""

import os
import sqlite3
import numpy as np
import pickle
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Import our compression pipeline
from compression_pipeline import CompressionPipeline
from compression_pipeline.performance import PerformanceOptimizer

# Optional imports for handling different media types
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - image processing will be limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - video processing will be limited")


class MultiMediaDatabaseCompressor:
    """
    Handles compression of mixed media database with different data types.
    """
    
    def __init__(self, db_path: str = "multimedia_compressed.db"):
        """Initialize the multimedia database compressor."""
        self.db_path = db_path
        self.conn = None
        
        # Different pipelines for different data types
        self.pipelines = {
            'image': CompressionPipeline(quality=0.8, block_size=8),      # Good for images
            'video': CompressionPipeline(quality=0.6, block_size=8),      # Lower quality for video
            'text': CompressionPipeline(quality=0.9, block_size=4),       # High quality for text
            'dataset': CompressionPipeline(quality=0.9, block_size=8),    # High quality for scientific data
            'binary': CompressionPipeline(quality=0.7, block_size=4),     # Medium quality for binary
            'default': CompressionPipeline(quality=0.7, block_size=8)     # Default pipeline
        }
        
        self._setup_database()
    
    def _setup_database(self):
        """Set up the database schema for multimedia data."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create table for multimedia data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multimedia_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                data_type TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                quality_metrics TEXT,
                compressed_blob BLOB NOT NULL,
                metadata TEXT,
                file_extension TEXT,
                processing_info TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON multimedia_data(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON multimedia_data(name)')
        
        self.conn.commit()
    
    def detect_data_type(self, data: Any, filename: str = "") -> str:
        """
        Automatically detect the type of data for optimal compression.
        
        Args:
            data: The data to analyze
            filename: Optional filename for extension-based detection
            
        Returns:
            Data type string for pipeline selection
        """
        # Check file extension first
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                return 'image'
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                return 'video'
            elif ext in ['.txt', '.md', '.csv', '.json', '.xml', '.html']:
                return 'text'
            elif ext in ['.bin', '.dat', '.exe', '.dll']:
                return 'binary'
        
        # Analyze data content
        if isinstance(data, str):
            return 'text'
        elif isinstance(data, bytes):
            return 'binary'
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Could be image or dataset
                if data.dtype in [np.uint8, np.int8] and np.max(data) <= 255:
                    return 'image'
                else:
                    return 'dataset'
            elif data.ndim == 1:
                return 'dataset'
            elif data.ndim == 3:
                return 'video'  # Treat 3D arrays as video frames
        elif isinstance(data, (list, tuple)):
            return 'dataset'
        
        return 'default'
    
    def process_image(self, image_data: Any, name: str) -> Tuple[Any, Dict]:
        """Process image data for compression."""
        processing_info = {'type': 'image', 'channels': 1}
        
        # Convert to numpy array if needed
        if isinstance(image_data, str):  # File path
            if PIL_AVAILABLE:
                img = Image.open(image_data)
                image_array = np.array(img)
                processing_info['original_format'] = img.format
                processing_info['original_mode'] = img.mode
            else:
                raise ValueError("PIL not available for image processing")
        else:
            image_array = np.array(image_data)
        
        # Handle different image formats
        if len(image_array.shape) == 3:  # RGB/RGBA
            processing_info['channels'] = image_array.shape[2]
            # Compress each channel separately
            compressed_channels = []
            for channel in range(image_array.shape[2]):
                channel_data = image_array[:, :, channel].astype(np.float32)
                compressed_channel = self.pipelines['image'].compress(channel_data)
                compressed_channels.append(compressed_channel)
            
            return compressed_channels, processing_info
        
        else:  # Grayscale
            image_array = image_array.astype(np.float32)
            compressed = self.pipelines['image'].compress(image_array)
            return compressed, processing_info
    
    def process_video(self, video_data: Any, name: str) -> Tuple[List, Dict]:
        """Process video data for compression."""
        processing_info = {'type': 'video', 'frames': 0}
        
        if isinstance(video_data, str):  # Video file path
            if not CV2_AVAILABLE:
                raise ValueError("OpenCV not available for video processing")
            
            cap = cv2.VideoCapture(video_data)
            compressed_frames = []
            frame_count = 0
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            processing_info.update({
                'fps': fps,
                'width': width,
                'height': height
            })
            
            print(f"Processing video: {width}x{height} @ {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for better compression
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Compress frame
                compressed_frame = self.pipelines['video'].compress(gray_frame.astype(np.float32))
                compressed_frames.append(compressed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    print(f"  Processed {frame_count} frames...")
            
            cap.release()
            processing_info['frames'] = frame_count
            
            return compressed_frames, processing_info
        
        else:  # Numpy array (3D video data)
            if len(video_data.shape) != 3:
                raise ValueError("Video data must be 3D array (frames, height, width)")
            
            compressed_frames = []
            for frame_idx in range(video_data.shape[0]):
                frame = video_data[frame_idx].astype(np.float32)
                compressed_frame = self.pipelines['video'].compress(frame)
                compressed_frames.append(compressed_frame)
            
            processing_info['frames'] = video_data.shape[0]
            return compressed_frames, processing_info
    
    def process_text(self, text_data: Any, name: str) -> Tuple[Any, Dict]:
        """Process text data for compression."""
        processing_info = {'type': 'text'}
        
        if isinstance(text_data, str):
            if os.path.isfile(text_data):  # File path
                with open(text_data, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:  # Direct text content
                content = text_data
        else:
            content = str(text_data)
        
        processing_info['length'] = len(content)
        processing_info['encoding'] = 'utf-8'
        
        compressed = self.pipelines['text'].compress(content)
        return compressed, processing_info
    
    def process_dataset(self, dataset_data: Any, name: str) -> Tuple[Any, Dict]:
        """Process numerical dataset for compression."""
        processing_info = {'type': 'dataset'}
        
        # Convert to numpy array
        if not isinstance(dataset_data, np.ndarray):
            dataset_array = np.array(dataset_data, dtype=np.float32)
        else:
            dataset_array = dataset_data.astype(np.float32)
        
        processing_info.update({
            'shape': dataset_array.shape,
            'dtype': str(dataset_array.dtype),
            'min_value': float(np.min(dataset_array)),
            'max_value': float(np.max(dataset_array)),
            'mean': float(np.mean(dataset_array))
        })
        
        compressed = self.pipelines['dataset'].compress(dataset_array)
        return compressed, processing_info
    
    def process_binary(self, binary_data: Any, name: str) -> Tuple[Any, Dict]:
        """Process binary data for compression."""
        processing_info = {'type': 'binary'}
        
        if isinstance(binary_data, str):  # File path
            with open(binary_data, 'rb') as f:
                content = f.read()
        else:
            content = bytes(binary_data)
        
        processing_info['size'] = len(content)
        
        compressed = self.pipelines['binary'].compress(content)
        return compressed, processing_info
    
    def store_data(self, name: str, data: Any, filename: str = "") -> Dict[str, Any]:
        """
        Store any type of data with automatic type detection and compression.
        
        Args:
            name: Unique identifier for the data
            data: Data to store (can be file path or actual data)
            filename: Optional filename for type detection
            
        Returns:
            Dictionary with storage information
        """
        print(f"\\n📦 Processing '{name}'...")
        
        # Detect data type
        data_type = self.detect_data_type(data, filename)
        print(f"   Detected type: {data_type}")
        
        # Calculate original size
        if isinstance(data, str) and os.path.isfile(data):
            original_size = os.path.getsize(data)
        elif hasattr(data, 'nbytes'):
            original_size = data.nbytes
        elif isinstance(data, (str, bytes)):
            original_size = len(data)
        else:
            original_size = len(str(data))
        
        # Process based on type
        start_time = time.time()
        
        try:
            if data_type == 'image':
                compressed_data, processing_info = self.process_image(data, name)
            elif data_type == 'video':
                compressed_data, processing_info = self.process_video(data, name)
            elif data_type == 'text':
                compressed_data, processing_info = self.process_text(data, name)
            elif data_type == 'dataset':
                compressed_data, processing_info = self.process_dataset(data, name)
            elif data_type == 'binary':
                compressed_data, processing_info = self.process_binary(data, name)
            else:
                compressed_data = self.pipelines['default'].compress(data)
                processing_info = {'type': 'default'}
            
            processing_time = time.time() - start_time
            processing_info['processing_time'] = processing_time
            
            # Serialize compressed data
            compressed_blob = pickle.dumps(compressed_data)
            compressed_size = len(compressed_blob)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO multimedia_data 
                (name, data_type, original_size, compressed_size, compression_ratio,
                 compressed_blob, metadata, file_extension, processing_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, data_type, original_size, compressed_size, compression_ratio,
                compressed_blob, json.dumps({}), Path(filename).suffix if filename else "",
                json.dumps(processing_info)
            ))
            
            self.conn.commit()
            
            # Return storage info
            storage_info = {
                'name': name,
                'data_type': data_type,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'space_saved_percent': (1 - compressed_size / original_size) * 100,
                'processing_time': processing_time
            }
            
            print(f"   ✅ Stored: {compression_ratio:.2f}x compression, "
                  f"{storage_info['space_saved_percent']:.1f}% space saved")
            
            return storage_info
            
        except Exception as e:
            print(f"   ❌ Error processing {name}: {str(e)}")
            raise
    
    def retrieve_data(self, name: str) -> Any:
        """Retrieve and decompress data from the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT compressed_blob, data_type, processing_info 
            FROM multimedia_data WHERE name = ?
        ''', (name,))
        
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Data '{name}' not found")
        
        compressed_blob, data_type, processing_info_json = result
        processing_info = json.loads(processing_info_json)
        
        # Deserialize compressed data
        compressed_data = pickle.loads(compressed_blob)
        
        # Decompress based on type
        pipeline = self.pipelines.get(data_type, self.pipelines['default'])
        
        if data_type == 'image' and isinstance(compressed_data, list):
            # Multi-channel image
            channels = []
            for compressed_channel in compressed_data:
                channel = pipeline.decompress(compressed_channel)
                channels.append(channel)
            reconstructed = np.stack(channels, axis=2)
        elif data_type == 'video' and isinstance(compressed_data, list):
            # Video frames
            frames = []
            for compressed_frame in compressed_data:
                frame = pipeline.decompress(compressed_frame)
                frames.append(frame)
            reconstructed = np.stack(frames, axis=0)
        else:
            # Single data item
            reconstructed = pipeline.decompress(compressed_data)
        
        return reconstructed
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        cursor = self.conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_items,
                SUM(original_size) as total_original_size,
                SUM(compressed_size) as total_compressed_size,
                AVG(compression_ratio) as avg_compression_ratio
            FROM multimedia_data
        ''')
        
        overall = cursor.fetchone()
        
        # Stats by data type
        cursor.execute('''
            SELECT 
                data_type,
                COUNT(*) as count,
                SUM(original_size) as original_size,
                SUM(compressed_size) as compressed_size,
                AVG(compression_ratio) as avg_ratio
            FROM multimedia_data
            GROUP BY data_type
        ''')
        
        by_type = cursor.fetchall()
        
        total_original = overall[1] or 0
        total_compressed = overall[2] or 0
        
        stats = {
            'total_items': overall[0],
            'total_original_size_mb': total_original / (1024 * 1024),
            'total_compressed_size_mb': total_compressed / (1024 * 1024),
            'total_space_saved_mb': (total_original - total_compressed) / (1024 * 1024),
            'overall_compression_ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'space_saved_percent': (1 - total_compressed / total_original) * 100 if total_original > 0 else 0,
            'by_data_type': {}
        }
        
        for row in by_type:
            data_type, count, orig_size, comp_size, avg_ratio = row
            stats['by_data_type'][data_type] = {
                'count': count,
                'original_size_mb': orig_size / (1024 * 1024),
                'compressed_size_mb': comp_size / (1024 * 1024),
                'avg_compression_ratio': avg_ratio,
                'space_saved_percent': (1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
            }
        
        return stats
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def create_sample_multimedia_data():
    """Create sample multimedia data for demonstration."""
    print("🎬 Creating sample multimedia data...")
    
    # Create sample directory
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Sample image
    image_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    if PIL_AVAILABLE:
        img = Image.fromarray(image_data)
        img.save("sample_data/sample_image.png")
    
    # 2. Sample video (as numpy array)
    video_data = np.random.randint(0, 255, (30, 64, 64), dtype=np.uint8)  # 30 frames
    np.save("sample_data/sample_video.npy", video_data)
    
    # 3. Sample text
    sample_text = """
    This is a sample text document for compression testing.
    It contains multiple lines and various characters.
    The compression pipeline should handle this efficiently.
    """ * 50  # Make it longer
    
    with open("sample_data/sample_text.txt", "w") as f:
        f.write(sample_text)
    
    # 4. Sample dataset
    dataset = np.random.rand(100, 50).astype(np.float32)
    np.save("sample_data/sample_dataset.npy", dataset)
    
    # 5. Sample binary
    binary_data = bytes(range(256)) * 10
    with open("sample_data/sample_binary.bin", "wb") as f:
        f.write(binary_data)
    
    print("   ✅ Sample data created in 'sample_data/' directory")


def main():
    """Main demonstration function."""
    print("🗄️ Multi-Media Database Compression Demo")
    print("=" * 60)
    
    # Create sample data
    create_sample_multimedia_data()
    
    # Initialize compressor
    compressor = MultiMediaDatabaseCompressor("multimedia_demo.db")
    
    # Sample data to compress
    multimedia_items = {
        # Direct data
        "text_content": "This is direct text content for compression testing! " * 100,
        "numpy_image": np.random.randint(0, 255, (64, 64), dtype=np.uint8),
        "dataset_2d": np.random.rand(50, 50).astype(np.float32),
        "time_series": np.sin(np.linspace(0, 10*np.pi, 1000)).astype(np.float32),
        "binary_content": bytes(range(128)) * 5,
        
        # File paths (if files exist)
        "sample_text_file": "sample_data/sample_text.txt",
        "sample_dataset_file": "sample_data/sample_dataset.npy",
        "sample_binary_file": "sample_data/sample_binary.bin",
    }
    
    # Add image file if PIL is available
    if PIL_AVAILABLE and os.path.exists("sample_data/sample_image.png"):
        multimedia_items["sample_image_file"] = "sample_data/sample_image.png"
    
    # Add video file if exists
    if os.path.exists("sample_data/sample_video.npy"):
        multimedia_items["sample_video_array"] = np.load("sample_data/sample_video.npy")
    
    # Store all items
    print("\\n📦 Storing multimedia items...")
    storage_results = {}
    
    for name, data in multimedia_items.items():
        try:
            if isinstance(data, str) and not os.path.exists(data):
                print(f"   ⚠️  Skipping {name}: File not found")
                continue
                
            result = compressor.store_data(name, data, filename=name if isinstance(data, str) else "")
            storage_results[name] = result
        except Exception as e:
            print(f"   ❌ Failed to store {name}: {e}")
    
    # Display statistics
    print("\\n📊 Database Statistics:")
    stats = compressor.get_database_stats()
    
    print(f"\\nOverall Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Original size: {stats['total_original_size_mb']:.2f} MB")
    print(f"  Compressed size: {stats['total_compressed_size_mb']:.2f} MB")
    print(f"  Space saved: {stats['total_space_saved_mb']:.2f} MB ({stats['space_saved_percent']:.1f}%)")
    print(f"  Overall compression ratio: {stats['overall_compression_ratio']:.2f}x")
    
    print(f"\\nBy Data Type:")
    for data_type, type_stats in stats['by_data_type'].items():
        print(f"  {data_type.upper()}:")
        print(f"    Count: {type_stats['count']}")
        print(f"    Avg ratio: {type_stats['avg_compression_ratio']:.2f}x")
        print(f"    Space saved: {type_stats['space_saved_percent']:.1f}%")
    
    # Test retrieval
    print("\\n🔄 Testing data retrieval...")
    test_items = list(storage_results.keys())[:3]  # Test first 3 items
    
    for item_name in test_items:
        try:
            print(f"   Retrieving '{item_name}'...")
            retrieved_data = compressor.retrieve_data(item_name)
            print(f"   ✅ Retrieved successfully: shape {getattr(retrieved_data, 'shape', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
    
    compressor.close()
    
    print("\\n🎉 Demo completed!")
    print("\\n💡 Key Benefits:")
    print("  - Automatic data type detection")
    print("  - Optimized compression for each data type")
    print("  - Unified database storage")
    print("  - Easy retrieval and decompression")
    print(f"  - Achieved {stats['overall_compression_ratio']:.2f}x overall compression")


if __name__ == "__main__":
    main()