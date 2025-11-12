#!/usr/bin/env python3
"""
Simple runner for the multimedia database compression demo.

This script makes it easy to test the compression pipeline with different types of data.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.getcwd())

try:
    from compression_pipeline import CompressionPipeline
    from multimedia_database_example import MultiMediaDatabaseCompressor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\\n💡 Make sure you're running from the project root directory")
    print("   and all dependencies are installed:")
    print("   pip install numpy scipy scikit-image psutil")
    sys.exit(1)


def quick_test():
    """Quick test with sample data."""
    print("🚀 Quick Multi-Media Compression Test")
    print("=" * 50)
    
    # Initialize compressor
    compressor = MultiMediaDatabaseCompressor("quick_test.db")
    
    # Create sample data of different types
    test_data = {
        # Text data
        "sample_text": "This is a sample text document. " * 100,
        
        # Image-like data (2D array)
        "sample_image": np.random.randint(0, 255, (64, 64), dtype=np.uint8),
        
        # Dataset (scientific data)
        "sample_dataset": np.random.rand(50, 30).astype(np.float32),
        
        # Time series data
        "time_series": np.sin(np.linspace(0, 4*np.pi, 500)).astype(np.float32),
        
        # Binary data
        "binary_data": bytes(range(256)),
        
        # Video-like data (3D array)
        "video_frames": np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8),
        
        # List data
        "list_data": [[i+j for j in range(10)] for i in range(10)],
    }
    
    print(f"\\n📦 Compressing {len(test_data)} different data types...")
    
    # Store all data
    results = {}
    for name, data in test_data.items():
        try:
            result = compressor.store_data(name, data)
            results[name] = result
            print(f"   ✅ {name}: {result['compression_ratio']:.2f}x compression")
        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")
    
    # Get statistics
    stats = compressor.get_database_stats()
    
    print(f"\\n📊 Results Summary:")
    print(f"   Total items compressed: {stats['total_items']}")
    print(f"   Overall compression ratio: {stats['overall_compression_ratio']:.2f}x")
    print(f"   Total space saved: {stats['space_saved_percent']:.1f}%")
    
    print(f"\\n🎯 Compression by data type:")
    for data_type, type_stats in stats['by_data_type'].items():
        print(f"   {data_type.upper()}: {type_stats['avg_compression_ratio']:.2f}x average")
    
    # Test retrieval
    print(f"\\n🔄 Testing retrieval...")
    test_item = list(results.keys())[0]
    try:
        retrieved = compressor.retrieve_data(test_item)
        print(f"   ✅ Successfully retrieved '{test_item}'")
        print(f"   📊 Retrieved data shape: {getattr(retrieved, 'shape', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Retrieval failed: {e}")
    
    compressor.close()
    return stats


def interactive_demo():
    """Interactive demo where user can add their own data."""
    print("\\n🎮 Interactive Demo")
    print("=" * 30)
    print("You can now add your own data to compress!")
    print("\\nSupported inputs:")
    print("  - File paths (images, videos, text files, etc.)")
    print("  - Direct text input")
    print("  - 'quit' to exit")
    
    compressor = MultiMediaDatabaseCompressor("interactive_demo.db")
    
    while True:
        print("\\n" + "-" * 30)
        user_input = input("Enter data (file path, text, or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # Generate a name for the data
        if os.path.isfile(user_input):
            name = Path(user_input).stem
            data = user_input  # File path
            print(f"📁 Processing file: {user_input}")
        else:
            name = f"text_input_{len(user_input)}"
            data = user_input  # Direct text
            print(f"📝 Processing text input ({len(user_input)} characters)")
        
        try:
            result = compressor.store_data(name, data, filename=user_input if os.path.isfile(user_input) else "")
            print(f"✅ Compressed successfully!")
            print(f"   Compression ratio: {result['compression_ratio']:.2f}x")
            print(f"   Space saved: {result['space_saved_percent']:.1f}%")
            print(f"   Data type: {result['data_type']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Final stats
    stats = compressor.get_database_stats()
    print(f"\\n📊 Final Statistics:")
    print(f"   Total items: {stats['total_items']}")
    print(f"   Overall compression: {stats['overall_compression_ratio']:.2f}x")
    
    compressor.close()


def main():
    """Main function with menu options."""
    print("🗄️ Multi-Media Database Compression")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\\nChoose a demo mode:")
        print("1. Quick test with sample data")
        print("2. Full demo with file examples")
        print("3. Interactive demo")
        
        choice = input("\\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            mode = "quick"
        elif choice == "2":
            mode = "full"
        elif choice == "3":
            mode = "interactive"
        else:
            mode = "quick"
    
    try:
        if mode == "quick":
            stats = quick_test()
            
        elif mode == "full":
            print("\\n🎬 Running full multimedia demo...")
            from multimedia_database_example import main as full_demo
            full_demo()
            
        elif mode == "interactive":
            interactive_demo()
            
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: quick, full, interactive")
            return
        
        print("\\n🎉 Demo completed successfully!")
        print("\\n💡 Next steps:")
        print("  - Check the generated .db files to see compressed data")
        print("  - Modify the code to use your own data")
        print("  - Integrate into your own database system")
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        print("\\n💡 Make sure all dependencies are installed:")
        print("   pip install numpy scipy scikit-image psutil")


if __name__ == "__main__":
    main()