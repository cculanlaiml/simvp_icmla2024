import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_numpy_file(file_path, cmap='viridis', save=False):
    array = np.load(file_path)

    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.title(file_path, fontsize=6)

    if save:
        output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.png"
        plt.savefig(output_filename)
        print(f"Image saved as {output_filename}")
    else:
        plt.show()


def pick_random_file(directory):
    numpy_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                numpy_files.append(os.path.join(root, file))

    if not numpy_files:
        raise ValueError("No numpy files found in the directory.")

    random_file = random.choice(numpy_files)
    print(f"No filename given - randomly chosen file: {random_file}")
    return random_file


def main():
    parser = argparse.ArgumentParser(description='Visualize NxN numpy arrays as images.')
    parser.add_argument('path', type=str, help='Path to a numpy file or directory.')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap to apply (default: viridis).')
    parser.add_argument('--save', action='store_true', help='Save the image instead of showing it.')

    args = parser.parse_args()

    if os.path.isfile(args.path):
        file_path = args.path
    elif os.path.isdir(args.path):
        file_path = pick_random_file(args.path)
    else:
        raise ValueError("The path provided is neither a file nor a directory.")

    visualize_numpy_file(file_path, cmap=args.cmap, save=args.save)


if __name__ == "__main__":
    main()