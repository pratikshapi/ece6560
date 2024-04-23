import numpy as np
import cv2
import os
from denoise_images import wavelet_denoising, gaussian_denoise
from denoise_pm import anisodiff_f1, anisodiff_f2

def psnr(target, ref):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mse(target, ref):
    return np.mean((target - ref) ** 2)

def calculate_metrics_for_denoised_images(noisy_image_path, original_image_path, Kf1=0.1, stepsf1=100, Kf2=4, stepsf2=50, wavelet='db4', wavelet_level=5, kernel_size=9, sigma=0.1):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Perona-Malik anisotropic diffusion
    denoised_image_pmf1 = anisodiff_f1(noisy_image, stepsf1, Kf1)
    denoised_image_pmf2 = anisodiff_f2(noisy_image, stepsf2, Kf2)
    psnr_pmf1 = psnr(original_image, denoised_image_pmf1)
    mse_pmf1 = mse(original_image, denoised_image_pmf1)
    psnr_pmf2 = psnr(original_image, denoised_image_pmf2)
    mse_pmf2 = psnr(original_image, denoised_image_pmf2)
    
    # Apply wavelet denoising
    denoised_image_wavelet = wavelet_denoising(noisy_image, wavelet, wavelet_level)
    psnr_wavelet = psnr(original_image, denoised_image_wavelet)
    mse_wavelet = mse(original_image, denoised_image_wavelet)
    
    # Apply Gaussian denoising
    denoised_image_gaussian = gaussian_denoise(noisy_image, kernel_size, sigma)
    psnr_gaussian = psnr(original_image, denoised_image_gaussian)
    mse_gaussian = mse(original_image, denoised_image_gaussian)
    
    # Extract the base name for the noisy image to construct the denoised filenames
    base_noisy_image_name = os.path.splitext(os.path.basename(noisy_image_path))[0]

    # Save the denoised images with the respective method name appended
    denoised_image_pmf1_path = f'denoised_images/{base_noisy_image_name}_pm_f1.jpg'
    denoised_image_pmf2_path = f'denoised_images/{base_noisy_image_name}_pm_f2.jpg'
    denoised_image_wavelet_path = f'denoised_images/{base_noisy_image_name}_wavelet.jpg'
    denoised_image_gaussian_path = f'denoised_images/{base_noisy_image_name}_gaussian.jpg'

    cv2.imwrite(denoised_image_pmf1_path, denoised_image_pmf1)
    cv2.imwrite(denoised_image_pmf2_path, denoised_image_pmf2)
    cv2.imwrite(denoised_image_wavelet_path, denoised_image_wavelet)
    cv2.imwrite(denoised_image_gaussian_path, denoised_image_gaussian)

    # Return metrics
    return {
        'psnr_pmf1': psnr_pmf1,
        'psnr_pmf2': psnr_pmf2,
        'psnr_wavelet': psnr_wavelet,
        'psnr_gaussian': psnr_gaussian,
        'mse_pmf1': mse_pmf1,
        'mse_pmf2': mse_pmf2,
        'mse_wavelet': mse_wavelet,
        'mse_gaussian': mse_gaussian
    }

if __name__ == '__main__':
    # Example usage:
    results_speckle = calculate_metrics_for_denoised_images('images/speckle_noise.jpg', 'images/original.jpg')
    results_gaussian = calculate_metrics_for_denoised_images('images/gaussian_noise.jpg', 'images/original.jpg')

    print("Results for speckle noise:")
    print(results_speckle)
    print("Results for Gaussian noise:")
    print(results_gaussian)

