import os
import numpy as np
from PIL import Image
import argparse

def jpeg_to_npz(input_folder, output_folder):
    """
    Transforms JPEG files in the input folder to .npz objects in the output folder.

    Args:
        input_folder (str): Path to the folder containing JPEG files.
        output_folder (str): Path to the folder where .npz files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all JPEG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)

            # Load the image
            with Image.open(input_path) as img:
                # Convert to NumPy array
                img_array = np.array(img)

            # Create output file path with the same name but .npz extension
            output_filename = os.path.splitext(filename)[0] + ".npz"
            output_path = os.path.join(output_folder, output_filename)

            # Save the NumPy array to .npz format
            np.savez_compressed(output_path, img=img_array)
            print(f"Converted {filename} to {output_filename}")

    print("Conversion completed.")

def main():
    parser = argparse.ArgumentParser(description="Convert JPEG files to NPZ format.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing JPEG files.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where NPZ files will be saved.")
    args = parser.parse_args()

    jpeg_to_npz(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
