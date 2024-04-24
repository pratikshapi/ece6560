import cv2
import numpy as np
import os
from denoise_pm import anisodiff_f1, anisodiff_f2
from get_errors import psnr, mse

def optimize_pmf_parameters(noisy_image_path, original_image_path, K_values, step_counts):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    best_psnr, best_mse, best_K, best_steps = -np.inf, np.inf, None, None
    
    for K in K_values:
        for steps in step_counts:
            denoised_image = anisodiff_f1(noisy_image, steps, K)
            current_psnr = psnr(original_image, denoised_image)
            current_mse = mse(original_image, denoised_image)
            if current_psnr > best_psnr: # and current_mse < best_mse:
                best_psnr, best_mse = current_psnr, current_mse
                best_K, best_steps = K, steps 
    
    return best_K, best_steps, best_psnr, best_mse

# Example K_values and step_counts for grid search
K_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5 , 1, 2, 3, 4]
step_counts = [5, 10, 50, 100, 150, 200]

# Run the optimization
best_K, best_steps, best_psnr, best_mse = optimize_pmf_parameters('images/speckle_noise.jpg', 'images/original.jpg', K_values, step_counts)
print(f"Best K: {best_K}, Best Steps: {best_steps}, Best PSNR: {best_psnr}, Best MSE: {best_mse}")
