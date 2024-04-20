import numpy as np
import pywt
import cv2

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(image, wavelet='db4', level=5):
    coeffs = pywt.wavedec2(image, wavelet, mode='per', level=level)
    sigma = (1/0.6745) * madev(coeffs[-1][0])  
    uthresh = sigma * np.sqrt(2 * np.log(image.size))

    # sigma = (1/0.6745) * madev(coeffs[-1][-1])  # Use the detail coefficients for estimation
    # uthresh = sigma * np.sqrt(2 * np.log(np.product(image.shape)))

    new_coeffs = [coeffs[0]] + [tuple(pywt.threshold(detail, value=uthresh, mode='hard') for detail in level) for level in coeffs[1:]]
    denoised_image = pywt.waverec2(new_coeffs, wavelet, mode='per')
    return denoised_image #np.clip(denoised_image, 0, 255).astype('uint8')

def gaussian_denoise(image, kernel_size=5, sigma=1.5):
    # Apply Gaussian blur to the image
    denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return denoised_image

noisy_image = cv2.imread('images/speckle_noise.jpg', cv2.IMREAD_GRAYSCALE)
denoised_image_wavelet = wavelet_denoising(noisy_image)
cv2.imwrite('denoised_images/speckle_noise_wavelet_denoise.jpg', denoised_image_wavelet)
denoised_image_gaussian = gaussian_denoise(noisy_image)
cv2.imwrite('denoised_images/speckle_noise_gaussian_denoise.jpg', denoised_image_gaussian)

noisy_image = cv2.imread('images/gaussian_noise.jpg', cv2.IMREAD_GRAYSCALE)
denoised_image_wavelet = wavelet_denoising(noisy_image)
cv2.imwrite('denoised_images/gaussian_noise_wavelet_denoise.jpg', denoised_image_wavelet)
denoised_image_gaussian = gaussian_denoise(noisy_image)
cv2.imwrite('denoised_images/gaussian_noise_gaussian_denoise.jpg', denoised_image_gaussian)