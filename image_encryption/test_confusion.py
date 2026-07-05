import os

from image_confusion import (
    load_image,
    display_image,
    print_image_info
)

from pixel_operations import show_pixel_information

print("Current Directory:", os.getcwd())

image = load_image("sample_images/flower.jpg")

print_image_info(image)

show_pixel_information(image)

display_image("Original Image", image)