import cv2


def show_pixel_information(image):
    """
    Display information about image pixels.
    """

    print("\n========== PIXEL INFORMATION ==========\n")

    # Image dimensions
    height, width, channels = image.shape

    print(f"Height : {height}")
    print(f"Width  : {width}")
    print(f"Channels : {channels}")

    # First pixel
    print("\nFirst Pixel (0,0):")
    print(image[0, 0])

    # Center pixel
    center_x = height // 2
    center_y = width // 2

    print("\nCenter Pixel:")
    print(image[center_x, center_y])

    # Last pixel
    print("\nLast Pixel:")
    print(image[height - 1, width - 1])

    print("\n=======================================\n")