import numpy as np
import cv2 as cv


def preprocessing(img):
  # Apply a Gaussian blur filter to reduce noise
  image_blurred = cv.GaussianBlur(img, (15, 15), 1, 1)
  # Define the kernel for the sharpening filter
  kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])
  # Apply the sharpening filter
  sharpened_image = cv.filter2D(image_blurred, -1, kernel)
  # Specify the window size for local binarization (must be an odd number)
  block_size = 71
  # Specify a constant to subtract from the local mean to calculate the threshold
  C = 2
  # Apply local binarization
  binary_image_1 = cv.adaptiveThreshold(sharpened_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)
  # Binarize the image based on a threshold
  threshold = np.mean(sharpened_image) * 0.9 # You can adjust this threshold according to your needs
  _, binary_image_2 = cv.threshold(sharpened_image, threshold, 255, cv.THRESH_BINARY)
  # Apply a mask to the original image to extract the desired objects
  result = cv.bitwise_or(binary_image_1, binary_image_2)
  color_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  color_image[:, :, 0] = result
  color_image[:, :, 1] = result
  color_image[:, :, 2] = result
  return color_image
