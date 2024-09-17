import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_images_from_subdir(subdir_path, num_images):
    images = []
    for i in range(num_images):
        file_path = os.path.join(subdir_path, f'{i}.npy')
        if os.path.exists(file_path):
            images.append(np.load(file_path))
        else:
            raise FileNotFoundError(f"File {file_path} not found.")
    return images

def create_image_grid(base_dir, subdir_name, cmap='viridis', save=False):
    input_dir = os.path.join(base_dir, 'inputs', subdir_name)
    true_dir = os.path.join(base_dir, 'trues', subdir_name)
    pred_dir = os.path.join(base_dir, 'preds', subdir_name)

    # Assuming that the number of images is the same in all subdirectories
    num_images = len(os.listdir(input_dir))

    input_images = load_images_from_subdir(input_dir, num_images)
    true_images = load_images_from_subdir(true_dir, num_images)
    pred_images = load_images_from_subdir(pred_dir, num_images)

    # Create a 3xN grid for visualization
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))

    for i in range(num_images):
        axes[0, i].imshow(input_images[i], cmap=cmap)
        axes[0, i].set_title(f'Input {i}')
        axes[0, i].axis('off')

        axes[1, i].imshow(true_images[i], cmap=cmap)
        axes[1, i].set_title(f'True {i}')
        axes[1, i].axis('off')

        axes[2, i].imshow(pred_images[i], cmap=cmap)
        axes[2, i].set_title(f'Pred {i}')
        axes[2, i].axis('off')

    plt.suptitle(f'Comparison for {subdir_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or show the image
    if save:
        output_filename = f"grid_{subdir_name}.png"
        plt.savefig(output_filename)
        print(f"Grid image saved as {output_filename}")
    else:
        plt.show()

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate a 3xN grid of images from input, true, and predicted directories.')
    parser.add_argument('base_dir', type=str, help='Base directory containing input, true, and pred subdirectories.')
    parser.add_argument('subdir_name', type=str, help='Name of the specific subdirectory within input/true/pred to process.')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap to apply (default: viridis).')
    parser.add_argument('--save', action='store_true', help='Save the grid image instead of showing it.')

    args = parser.parse_args()

    # Generate the image grid
    create_image_grid(args.base_dir, args.subdir_name, cmap=args.cmap, save=args.save)

if __name__ == "__main__":
    main()