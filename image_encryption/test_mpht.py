from image_confusion import *
from chaos import SkewTentMap
from mpht_shuffle import MPHTShuffle

print("Loading Image...")

image = load_image("sample_images/flower.jpg")

print_image_info(image)

height, width, channels = image.shape

number_of_pixels = height * width

print("\nGenerating Chaos Sequence...")

chaos = SkewTentMap(
    x0=0.45,
    p=0.35
)

sequence = chaos.generate_sequence(number_of_pixels)

print("Sequence Generated.")

print("\nShuffling Image...")

mpht = MPHTShuffle(sequence)

shuffled_image, permutation = mpht.shuffle(image)

print("Image Successfully Shuffled.")

display_image("Original", image)

display_image("MPHT Shuffled", shuffled_image)

save_image(
    "encrypted/shuffled_flower.png",
    shuffled_image
)

print("\nSaved in encrypted/shuffled_flower.png")