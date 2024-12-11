import cv2
import numpy as np

def calculate_reconstruction_error(image_path_rec, image_path_gt):
    
    # Load the images in grayscale
    reconstructed = cv2.imread(image_path_rec, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(image_path_gt, cv2.IMREAD_GRAYSCALE)

    if reconstructed is None or ground_truth is None:
        raise ValueError("One or both image paths are invalid or the images could not be loaded.")

    # Ensure both images have the same dimensions
    if reconstructed.shape != ground_truth.shape:
        raise ValueError("Reconstructed and ground truth images must have the same dimensions.")

    # Calculate absolute difference
    absolute_difference = np.abs(reconstructed - ground_truth)

    # Calculate sum of absolute differences
    numerator = np.sum(absolute_difference)

    # Calculate sum of ground truth pixel values
    denominator = np.sum(ground_truth)

    # Compute reconstruction error as a percentage
    reconstruction_error = (numerator / denominator) * 100

    return reconstruction_error

# Example usage
if __name__ == "__main__":
    # Paths to the reconstructed and ground truth images
    reconstructed_image_path = "/content/sample_hr_input.png"
    ground_truth_image_path = "/content/sample_sr_output.png"

    try:
        error = calculate_reconstruction_error(reconstructed_image_path, ground_truth_image_path)
        print(f"Reconstruction Error: {error:.2f}%")
    except ValueError as e:
        print(e)
