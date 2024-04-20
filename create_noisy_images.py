import numpy as np
import cv2

def add_speckle_noise(image, mean=0, var=0.1):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)    
    noisy = image + image * gauss * var
    return noisy.astype('uint8')

def add_speckle_noise_bw(image, mean=0, var=0.1):
    row, col = image.shape
    gauss = np.random.randn(row, col)
    noisy = image + image * gauss * var
    return noisy.astype('uint8')

def add_gaussian_noise(image, mean=0, var=0.1):
    row, col = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col)) * 50 
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)  # To limit pixel values between 0 and 255
    return noisy.astype('uint8')

image_bw = cv2.imread('images/dog.jpg', cv2.IMREAD_GRAYSCALE)

# Add speckle noise
noisy_image_speckle_bw = add_speckle_noise_bw(image_bw)
cv2.imwrite('images/speckle_noise.jpg', noisy_image_speckle_bw)

# Add Gaussian noise
noisy_image_gaussian = add_gaussian_noise(image_bw)
cv2.imwrite('images/gaussian_noise.jpg', noisy_image_gaussian)

# # Display or save the noisy image
# cv2.imshow('Speckle Noise Image', noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
