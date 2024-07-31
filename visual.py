import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_binary_file(file_path, rows, cols):
    """
    Reads a binary file containing doubles (float64) and returns a 2D numpy array.
    
    Parameters:
        file_path (str): Path to the binary file.
        rows (int): Number of rows in the 2D array.
        cols (int): Number of columns in the 2D array.
        
    Returns:
        np.ndarray: 2D array with shape (rows, cols).
    """
    # Define the data type for doubles (float64)
    dtype = np.float64

    # Read the binary file
    data = np.fromfile(file_path, dtype=dtype)

    # Reshape the data into the desired 2D shape
    array_2d = data.reshape((rows, cols))
    
    return array_2d

def visualize_2d_array(array_2d, filename):
    """
    Visualizes a 2D numpy array as a heatmap.
    
    Parameters:
        array_2d (np.ndarray): 2D numpy array to visualize.
    """
    plt.imshow(array_2d, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('2D Array Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.savefig(filename)
    plt.close()

# Example usage


parser = argparse.ArgumentParser(description='Process some options and flags.')

parser.add_argument('-i', '--input', type=str, help='Input binary file')
parser.add_argument('-o', '--output', type=str, help='output file')
parser.add_argument('-r', '--rows', type=int, default=1, help='rows')
parser.add_argument('-c', '--columns', type=int, default=1, help='columns')


# Parse arguments
args = parser.parse_args()

file_path = args.input  # Path to your binary file
rows = args.rows  # Number of rows in the 2D array
cols = args.columns  # Number of columns in the 2D array

# Read and visualize the 2D array
array_2d = read_binary_file(file_path, rows, cols)
visualize_2d_array(array_2d, args.output)