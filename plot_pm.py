import numpy as np
import matplotlib.pyplot as plt
import cv2
from denoise_pm import anisodiff_f1, anisodiff_f2, F1, F2

K_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]

def apply_anisodiff_functions(img, steps):
    diffused_images_f1 = [anisodiff_f1(img, steps, K) for K in K_values]
    diffused_images_f2 = [anisodiff_f2(img, steps, K) for K in K_values]
    return diffused_images_f1, diffused_images_f2

def plot_diffusion_coefficient_vs_gradient_line(F, equation_number):
    image_gradients = np.linspace(0, 10, 100)
    
    for K in K_values:
        diffusion = F(image_gradients, K)
        plt.plot(image_gradients, diffusion, label=f'K = {K}')
    
    plt.title(f'Effect on diffusion by changing K (given by equation {equation_number})')
    plt.xlabel('Image Gradient')
    plt.ylabel('Diffusion')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diffusion_coefficient_vs_gradient_line_and_save(F, equation_number, filename):
    image_gradients = np.linspace(0, 10, 100)
    plt.figure()

    for K in K_values:
        diffusion = F(image_gradients, K)
        plt.plot(image_gradients, diffusion, label=f'K = {K}')
    
    plt.title(f'Effect on diffusion by changing K (given by equation {equation_number})')
    plt.xlabel('Image Gradient')
    plt.ylabel('Diffusion')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'{filename}_equation_{equation_number}.png')
    plt.close()  # Close the figure to free memory

# Example usage:
img = cv2.imread('images/gaussian_noise.jpg', cv2.IMREAD_GRAYSCALE)
steps = 10
f1_images, f2_images = apply_anisodiff_functions(img, steps)

# Now call the plotting function for each F function and save the plots
plot_diffusion_coefficient_vs_gradient_line_and_save(F1, 3, 'F1_plot')
plot_diffusion_coefficient_vs_gradient_line_and_save(F2, 4, 'F2_plot')

# # Now call the plotting function for each F function
# plot_diffusion_coefficient_vs_gradient_line(F1, 3)  # For equation 3
# plot_diffusion_coefficient_vs_gradient_line(F2, 4)  # For equation 4
