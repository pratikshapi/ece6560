import pywt
import numpy as np
import cv2

def denoise_image_wavelet(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(image.size)) * 5
    coeffs[1:] = [tuple(pywt.threshold(detail, threshold, mode='soft') for detail in level) for level in coeffs[1:]]
    image_denoised = pywt.waverec2(coeffs, wavelet)
    return np.clip(image_denoised, 0, 255).astype('uint8')

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='db2', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


noisy_image = cv2.imread('images/speckle_noisy_image.jpg', cv2.IMREAD_GRAYSCALE)
denoised_image = wavelet_denoising(noisy_image)
cv2.imwrite('denoised_images/denoised_image_wavelet.jpg', denoised_image)



# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import warnings
import matplotlib.pylab as pyp
import math
from PIL import Image

"""
    Anisotropic diffusion.
    
    Cian Conway - 10126767
    Patrick Stapleton - 10122834
    Ivan McCaffrey - 10098119
    
"""


resultImage = np.array(Image.open('assets/noisy-lena.png').convert('L'))  ## Specify the original file path

im_min, im_max = resultImage.min(), resultImage.max()

print("Original image:", resultImage.shape, resultImage.dtype, im_min, im_max)
resultImage = (resultImage - im_min) / (float)(im_max - im_min)   ## Conversion
print("Perona-Malik Anisotropic Diffusion:", resultImage.shape, resultImage.dtype, resultImage.min(), resultImage.max())

pyp.figure('Image BEFORE Perona-Malik anisotropic diffusion')
pyp.imshow(resultImage, cmap='gray')
pyp.axis('on')


pyp.show() 

"""

	****Stopping Functions*****
	Alternate Stopping Function
	def f(lam,b): 
		func = 1/(1 + ((lam/b)**2))
		return func
	
"""
def f(lam,b):
    return np.exp(-1* (np.power(lam,2))/(np.power(b,2)))

def anisodiff(im, steps, b, lam = 0.25):  #takes image input, the number of iterations, 
    

    im_new = np.zeros(im.shape, dtype=im.dtype) 
    for t in range(steps): 
        dn = im[:-2,1:-1] - im[1:-1,1:-1] 
        ds = im[2:,1:-1] - im[1:-1,1:-1] 
        de = im[1:-1,2:] - im[1:-1,1:-1] 
        dw = im[1:-1,:-2] - im[1:-1,1:-1] 
        im_new[1:-1,1:-1] = im[1:-1,1:-1] +\
                            lam * (f(dn,b)*dn + f (ds,b)*ds + 
                                    f (de,b)*de + f (dw,b)*dw) 
        im = im_new 
    return im
  

im2 = anisodiff(resultImage, 60, 0.15, 0.15)
pyp.figure('Image AFTER Perona-Malik anisotropic diffusion')
pyp.imshow(im2, cmap='gray')
pyp.axis('on')
pyp.show()







# plot_pm_bkp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from denoise_pm import anisodiff_f1, anisodiff_f2, F1, F2

def apply_anisodiff_functions(img, steps):
    K_values = [0.1, 0.5, 1, 2]
    diffused_images_f1 = [anisodiff_f1(img, steps, K) for K in K_values]
    diffused_images_f2 = [anisodiff_f2(img, steps, K) for K in K_values]
    return diffused_images_f1, diffused_images_f2

def plot_diffusion_vs_gradient():
    K_values = np.linspace(0.1, 2, 100)  # More refined range of K values
    image_gradient = np.linspace(-7, 7, 100)  # Simulated image gradient range

    F1_results = [F1(dt, K) for K in K_values for dt in image_gradient]
    F2_results = [F2(dt, K) for K in K_values for dt in image_gradient]

    plt.figure(figsize=(10, 5))

    # Plot for F1
    plt.subplot(1, 2, 1)
    plt.title('F1: Diffusion vs Gradient')
    plt.imshow(np.reshape(F1_results, (100, 100)), extent=[-7, 7, 0.1, 2], aspect='auto')
    plt.xlabel('Image Gradient')
    plt.ylabel('K value')
    plt.colorbar()

    # Plot for F2
    plt.subplot(1, 2, 2)
    plt.title('F2: Diffusion vs Gradient')
    plt.imshow(np.reshape(F2_results, (100, 100)), extent=[-7, 7, 0.1, 2], aspect='auto')
    plt.xlabel('Image Gradient')
    plt.ylabel('K value')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_diffusion_vs_gradient_line(F, K_values, max_gradient=10):
    image_gradients = np.linspace(0, max_gradient, 100)
    plt.figure()

    for K in K_values:
        diffusion = F(image_gradients, K)
        plt.plot(image_gradients, diffusion, label=f'K = {K}')

    plt.title(f'Effect on diffusion by changing K (given by equation {3 if F == F1 else 4})')
    plt.xlabel('Image Gradient')
    plt.ylabel('Diffusion')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
K_values = [0.1, 0.5, 1, 2]

# Example usage:
img = cv2.imread('images/gaussian_noise.jpg', cv2.IMREAD_GRAYSCALE)  
f1_images, f2_images = apply_anisodiff_functions(img, steps=10)
plot_diffusion_vs_gradient(F1, K_values)  # For equation 3
plot_diffusion_vs_gradient(F2, K_values)  # For equation 4