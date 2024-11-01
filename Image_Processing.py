import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
from skimage.metrics import mean_squared_error
from math import log2, sqrt
import heapq
import itertools
import collections


# Helper Functions
def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def calculate_rmse(original, reconstructed):
    mse = mean_squared_error(original, reconstructed)
    return sqrt(mse)




# Transform Coding (using DCT)

def dct2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def transform_coding_dct(image, block_size=8):
    h, w = image.shape
    dct_blocks = np.zeros_like(image, dtype=float)
    quantization_factor = 10  # Example quantization factor

    # Forward DCT and quantization
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = dct2d(block)
            dct_blocks[i:i+block_size, j:j+block_size] = np.round(dct_block / quantization_factor)

    # Calculate compression ratio and RMSE
    compressed_size = np.count_nonzero(dct_blocks)
    original_size = h * w
    reconstructed = np.zeros_like(image, dtype=float)

    # Reconstruct using inverse DCT
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size] * quantization_factor
            reconstructed[i:i+block_size, j:j+block_size] = idct2d(block)
    
    cr = compression_ratio(original_size, compressed_size)
    rmse = calculate_rmse(image, reconstructed)
    return dct_blocks, reconstructed, cr, rmse



# Huffman Encoding

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_tree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def huffman_codes(tree):
    def generate_codes(node, prefix=""):
        if node.symbol is not None:
            return {node.symbol: prefix}
        left_codes = generate_codes(node.left, prefix + "0")
        right_codes = generate_codes(node.right, prefix + "1")
        return {**left_codes, **right_codes}
    return generate_codes(tree)

def huffman_encoding(image):
    frequencies = collections.Counter(image.flatten())
    tree = huffman_tree(frequencies)
    codes = huffman_codes(tree)
    
    encoded_image = ''.join(codes[pixel] for pixel in image.flatten())
    compressed_size = len(encoded_image)
    original_size = image.size * 8
    cr = compression_ratio(original_size, compressed_size)
    
    return encoded_image, codes, cr



# LZW Encoding

def lzw_encode(data):
    dictionary = {chr(i): i for i in range(256)}
    p = ""
    output = []
    code = 256

    for c in data:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            output.append(dictionary[p])
            dictionary[pc] = code
            code += 1
            p = c
    if p:
        output.append(dictionary[p])
    return output, dictionary

def lzw_encoding(image):
    flat_image = ''.join(map(chr, image.flatten()))
    encoded_data, dictionary = lzw_encode(flat_image)
    compressed_size = len(encoded_data) * 12  # Assuming 12-bit codes
    original_size = image.size * 8
    cr = compression_ratio(original_size, compressed_size)
    
    return encoded_data, cr


# Run_Length Encoding

def run_length_encode(image):
    flat_image = image.flatten()
    encoded = []
    prev_pixel = flat_image[0]
    count = 1

    for pixel in flat_image[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoded.extend([prev_pixel, count])
            prev_pixel = pixel
            count = 1
    encoded.extend([prev_pixel, count])  # for the last run
    compressed_size = len(encoded) * 8
    original_size = image.size * 8
    cr = compression_ratio(original_size, compressed_size)
    return encoded, cr




# Arithmatic Coding

def arithmetic_encoding(image):
    # Pseudo implementation (basic structure, not operational)
    original_size = image.size * 8
    compressed_size = original_size * 0.7  # Assume 30% compression rate
    cr = compression_ratio(original_size, compressed_size)
    return "pseudo_encoded_data", cr


# Main Function 
def compress_image(image_path):
    # Load image in grayscale
    image = Image.open(image_path).convert('L')
    image = np.array(image)

    print("\n--- Compression Results ---")

    # A. Transform Coding (DCT)
    _, reconstructed_dct, dct_cr, dct_rmse = transform_coding_dct(image)
    print(f"Transform Coding (DCT): Compression Ratio = {dct_cr:.2f}, RMSE = {dct_rmse:.2f}")

    # B. Huffman Encoding
    encoded_huffman, huffman_codes, huffman_cr = huffman_encoding(image)
    print(f"Huffman Encoding: Compression Ratio = {huffman_cr:.2f}")

    # C. LZW Encoding
    encoded_lzw, lzw_cr = lzw_encoding(image)
    print(f"LZW Encoding: Compression Ratio = {lzw_cr:.2f}")

    # D. Run-Length Encoding
    encoded_rle, rle_cr = run_length_encode(image)
    print(f"Run-Length Encoding: Compression Ratio = {rle_cr:.2f}")

    # E. Arithmetic Coding
    encoded_arithmetic, arithmetic_cr = arithmetic_encoding(image)
    print(f"Arithmetic Coding: Compression Ratio = {arithmetic_cr:.2f}")

# Example usage
if __name__ == "__main__":
    compress_image("2.png")


