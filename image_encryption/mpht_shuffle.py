import numpy as np


class MPHTShuffle:
    """
    MPHT-based pixel permutation using a chaotic sequence.
    """

    def __init__(self, chaos_sequence):
        self.chaos_sequence = chaos_sequence

    def shuffle(self, image):
        """
        Shuffle the image pixels using the chaotic sequence.
        """

        # Get image dimensions
        height, width, channels = image.shape

        # Flatten image into N x 3 array
        flat_pixels = image.reshape(-1, channels)

        total_pixels = flat_pixels.shape[0]

        # Check sequence length
        if len(self.chaos_sequence) != total_pixels:
            raise ValueError(
                f"Sequence length ({len(self.chaos_sequence)}) "
                f"must equal number of pixels ({total_pixels})"
            )

        # Sort chaos values and get permutation indices
        permutation = np.argsort(self.chaos_sequence)

        # Rearrange pixels
        shuffled_pixels = flat_pixels[permutation]

        # Reshape back
        shuffled_image = shuffled_pixels.reshape(height, width, channels)

        return shuffled_image, permutation