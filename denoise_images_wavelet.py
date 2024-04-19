import pywt
import numpy as np
import cv2

def denoise_image_wavelet(image, wavelet='db1', level=1):
    # Decompose to wavelet coefficients
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Thresholding detail coefficients
    threshold = np.sqrt(2 * np.log(image.size)) * 0.04  # Adjust threshold value as needed
    coeffs[1:] = [tuple(pywt.threshold(detail, threshold, mode='soft') for detail in level) for level in coeffs[1:]]
    
    # Reconstruct the image
    image_denoised = pywt.waverec2(coeffs, wavelet)
    return np.clip(image_denoised, 0, 255).astype('uint8')

# def denoise_image_wavelet(image, wavelet='db1', level=1):
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#     coeffs_H = list(coeffs)   
#     coeffs_H[0] *= 0  # Set the approximation coefficients to zero

#     threshold = np.sqrt(2 * np.log(image.size)) * 0.04
#     coeffs[1:] = [(pywt.threshold(i, threshold, mode='soft') if i is not None else None) for i in coeffs[1:]]

#     # new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    
#     # Reconstruct the image
#     image_denoised = pywt.waverec2(coeffs, wavelet)
#     return np.clip(image_denoised, 0, 255).astype('uint8')
    
#     # # Thresholding details coefficients
#     # for i in range(1, len(coeffs_H)):
#     #     coeffs_H[i] = tuple(component * 0 for component in coeffs_H[i])
        
#     # # Reconstructing the image
#     # image_denoised = pywt.waverec2(coeffs_H, wavelet)
#     # return np.clip(image_denoised, 0, 255).astype('uint8')

noisy_image = cv2.imread('images/gaussian_noisy_image.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Noisy Image', noisy_image)
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

denoised_image = denoise_image_wavelet(noisy_image)
cv2.imwrite('denoised_images/denoised_image_wavelet.jpg', denoised_image)
