from image_confusion import (
    load_image,
    display_image,
    save_image,
    print_image_info
)

from chaos import SkewTentMap
from mpht_shuffle import MPHTShuffle

print("Loading Image...")

image = load_image("sample_images/flower.jpg")

print_image_info(image)

height, width, channels = image.shape

total_pixels = height * width

print("\nGenerating Chaos Sequence...")

chaos = SkewTentMap(
    x0=0.417,
    p=0.381
)

sequence = chaos.generate_sequence(total_pixels)

print("Sequence Generated.")

print("\nShuffling Image...")

mpht = MPHTShuffle(sequence)

shuffled_image, permutation = mpht.shuffle(image)

print("Image Successfully Shuffled.")

print("\nChecking Pixels")

print("Original:", image[0, 0])

print("Shuffled:", shuffled_image[0, 0])

display_image("Original Image", image)

display_image("Shuffled Image", shuffled_image)

save_image(
    "encrypted/shuffled_flower.png",
    shuffled_image
)

print("\nSaved Successfully.")