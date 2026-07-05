import numpy as np


class MPHTShuffle:
    """
    MPHT-style image confusion using a chaotic sequence.
    """

    def __init__(self, chaos_sequence):
        self.chaos_sequence = chaos_sequence

    def shuffle(self, image):

        height, width, channels = image.shape

        # Flatten image
        pixels = image.reshape(-1, channels)

        total_pixels = len(pixels)

        if len(self.chaos_sequence) != total_pixels:
            raise ValueError("Chaos sequence length mismatch.")

        # Convert chaos values into integer indices
        indices = np.arange(total_pixels)

        # Multiply by a large number and sort
        random_order = np.argsort(self.chaos_sequence * 1000000)

        # Shuffle pixels
        shuffled_pixels = pixels[random_order]

        # Reshape back
        shuffled_image = shuffled_pixels.reshape(height, width, channels)

        return shuffled_image, random_order