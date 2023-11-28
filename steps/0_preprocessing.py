import os
import cv2
from utilsPy.utils_preprocessing import preprocessing

# Input folder path containing images
input_folder = "data/images"

# Output folder path to save preprocessed images
os.makedirs("data/binary_images", exist_ok=True)
output_folder = "data/binary_images"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        preprocessed_img = preprocessing(img)
        cv2.imwrite(os.path.join(output_folder, filename), preprocessed_img)
        