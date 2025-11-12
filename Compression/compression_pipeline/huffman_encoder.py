"""
Huffman encoding component for the compression pipeline.

This module implements Huffman encoding and decoding with frequency table building
and Huffman tree construction for optimal compression.
"""

import heapq
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import Counter
from .models import HuffmanNode


class HuffmanEncoder:
    """Implements Huffman encoding and decoding operations."""
    
    def __init__(self):
        """Initialize Huffman encoder."""
        self.code_table: Optional[Dict[int, str]] = None
        self.decode_table: Optional[Dict[str, int]] = None
    
    def encode(self, data: np.ndarray) -> Tuple[bytes, Dict[int, str]]:
        """
        Encode data using Huffman coding.
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Tuple of (encoded_bytes, code_table)
            
        Raises:
            ValueError: If data is empty or contains invalid values
        """
        if data.size == 0:
            raise ValueError("Cannot encode empty data")
        
        # Flatten data and convert to integers
        flat_data = data.flatten().astype(np.int32)
        
        # Build frequency table
        frequencies = self._build_frequency_table(flat_data)
        
        if len(frequencies) == 0:
            raise ValueError("No data to encode")
        
        # Handle single symbol case
        if len(frequencies) == 1:
            symbol = list(frequencies.keys())[0]
            code_table = {symbol: '0'}
            # Encode as repeated '0' bits
            bit_string = '0' * len(flat_data)
        else:
            # Build Huffman tree and generate codes
            root = self._build_huffman_tree(frequencies)
            code_table = self._generate_codes(root)
            
            # Encode data
            bit_string = ''.join(code_table[symbol] for symbol in flat_data)
        
        # Convert bit string to bytes
        encoded_bytes = self._bits_to_bytes(bit_string)
        
        # Store code table for decoding
        self.code_table = code_table
        self.decode_table = {code: symbol for symbol, code in code_table.items()}
        
        return encoded_bytes, code_table
    
    def decode(self, encoded_data: bytes, code_table: Dict[int, str], 
               original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decode Huffman encoded data.
        
        Args:
            encoded_data: Encoded data as bytes
            code_table: Huffman code table for decoding
            original_shape: Shape to restore the decoded data
            
        Returns:
            Decoded data as numpy array
            
        Raises:
            ValueError: If code table is invalid or decoding fails
        """
        if not code_table:
            raise ValueError("Code table cannot be empty")
        
        # Create reverse lookup table
        decode_table = {code: symbol for symbol, code in code_table.items()}
        
        # Convert bytes back to bit string
        bit_string = self._bytes_to_bits(encoded_data)
        
        # Decode bit string
        decoded_symbols = self._decode_bit_string(bit_string, decode_table, 
                                                 int(np.prod(original_shape)))
        
        # Reshape to original shape
        decoded_array = np.array(decoded_symbols, dtype=np.int32).reshape(original_shape)
        
        return decoded_array
    
    def _build_frequency_table(self, data: np.ndarray) -> Dict[int, int]:
        """
        Build frequency table for Huffman tree construction.
        
        Args:
            data: Flattened input data
            
        Returns:
            Dictionary mapping symbols to their frequencies
        """
        # Use Counter for efficient frequency counting
        frequencies = Counter(data.tolist())
        return dict(frequencies)
    
    def _build_huffman_tree(self, frequencies: Dict[int, int]) -> HuffmanNode:
        """
        Construct Huffman tree from frequency table.
        
        Args:
            frequencies: Dictionary mapping symbols to frequencies
            
        Returns:
            Root node of the Huffman tree
            
        Raises:
            ValueError: If frequencies dictionary is empty
        """
        if not frequencies:
            raise ValueError("Frequencies dictionary cannot be empty")
        
        # Create leaf nodes and add to priority queue
        heap = []
        for symbol, freq in frequencies.items():
            node = HuffmanNode(value=symbol, frequency=freq)
            heapq.heappush(heap, node)
        
        # Build tree by combining nodes
        while len(heap) > 1:
            # Get two nodes with lowest frequency
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create internal node
            merged = HuffmanNode(
                value=None,
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def _generate_codes(self, root: HuffmanNode) -> Dict[int, str]:
        """
        Generate Huffman codes from tree.
        
        Args:
            root: Root node of Huffman tree
            
        Returns:
            Dictionary mapping symbols to their binary codes
        """
        if root.is_leaf():
            # Single symbol case
            return {root.value: '0'}
        
        codes = {}
        
        def traverse(node: HuffmanNode, code: str = ''):
            if node.is_leaf():
                codes[node.value] = code
            else:
                if node.left:
                    traverse(node.left, code + '0')
                if node.right:
                    traverse(node.right, code + '1')
        
        traverse(root)
        return codes
    
    def _bits_to_bytes(self, bit_string: str) -> bytes:
        """
        Convert bit string to bytes with padding information.
        
        Args:
            bit_string: String of '0' and '1' characters
            
        Returns:
            Encoded bytes with padding info in first byte
        """
        # Calculate padding needed
        padding = (8 - len(bit_string) % 8) % 8
        
        # Add padding
        padded_bits = bit_string + '0' * padding
        
        # Convert to bytes
        byte_array = bytearray()
        
        # First byte stores padding information
        byte_array.append(padding)
        
        # Convert bit string to bytes
        for i in range(0, len(padded_bits), 8):
            byte_chunk = padded_bits[i:i+8]
            byte_value = int(byte_chunk, 2)
            byte_array.append(byte_value)
        
        return bytes(byte_array)
    
    def _bytes_to_bits(self, data: bytes) -> str:
        """
        Convert bytes back to bit string, removing padding.
        
        Args:
            data: Encoded bytes
            
        Returns:
            Bit string without padding
        """
        if len(data) < 1:
            return ''
        
        # First byte contains padding information
        padding = data[0]
        
        # Convert remaining bytes to bit string
        bit_string = ''
        for byte_val in data[1:]:
            bit_string += format(byte_val, '08b')
        
        # Remove padding
        if padding > 0:
            bit_string = bit_string[:-padding]
        
        return bit_string
    
    def _decode_bit_string(self, bit_string: str, decode_table: Dict[str, int], 
                          expected_length: int) -> List[int]:
        """
        Decode bit string using decode table.
        
        Args:
            bit_string: Encoded bit string
            decode_table: Mapping from codes to symbols
            expected_length: Expected number of decoded symbols
            
        Returns:
            List of decoded symbols
            
        Raises:
            ValueError: If decoding fails or produces wrong length
        """
        decoded_symbols = []
        current_code = ''
        
        for bit in bit_string:
            current_code += bit
            
            if current_code in decode_table:
                decoded_symbols.append(decode_table[current_code])
                current_code = ''
                
                # Stop if we've decoded enough symbols
                if len(decoded_symbols) == expected_length:
                    break
        
        # Check if we have leftover bits (shouldn't happen with valid encoding)
        if current_code and len(decoded_symbols) < expected_length:
            raise ValueError(f"Invalid encoding: leftover bits '{current_code}'")
        
        if len(decoded_symbols) != expected_length:
            raise ValueError(f"Decoded {len(decoded_symbols)} symbols, expected {expected_length}")
        
        return decoded_symbols
    
    def calculate_compression_ratio(self, original_data: np.ndarray, 
                                  encoded_data: bytes) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_data: Original uncompressed data
            encoded_data: Huffman encoded data
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        original_size = original_data.nbytes
        compressed_size = len(encoded_data)
        
        if compressed_size == 0:
            return float('inf')
        
        return original_size / compressed_size
    
    def get_code_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the current code table.
        
        Returns:
            Dictionary with code statistics
        """
        if not self.code_table:
            return {}
        
        code_lengths = [len(code) for code in self.code_table.values()]
        
        return {
            'num_symbols': len(self.code_table),
            'min_code_length': min(code_lengths) if code_lengths else 0,
            'max_code_length': max(code_lengths) if code_lengths else 0,
            'avg_code_length': sum(code_lengths) / len(code_lengths) if code_lengths else 0
        }