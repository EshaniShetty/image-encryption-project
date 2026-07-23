import numpy as np


class ChaosDiffusionV2:

    def __init__(self, chaos_sequence):
        self.chaos_sequence = chaos_sequence

    def diffuse(self, image):

        flat = image.flatten().astype(np.int32)
        N = len(flat)

        chaos = (
            np.floor(
                np.array(self.chaos_sequence[:N]) * 256
            ).astype(np.int32)
        )

        cipher = np.zeros_like(flat)

        # First pixel
        cipher[0] = (flat[0] + chaos[0]) % 256

        # Forward adaptive diffusion
        for i in range(1, N):

            key = (
                chaos[i]
                + cipher[i - 1]
                + ((i * 31) % 256)
            ) % 256

            cipher[i] = (
                flat[i]
                + key
            ) % 256

        return cipher.astype(np.uint8).reshape(image.shape)

    def inverse_diffuse(self, image):

        flat = image.flatten().astype(np.int32)
        N = len(flat)

        chaos = (
            np.floor(
                np.array(self.chaos_sequence[:N]) * 256
            ).astype(np.int32)
        )

        plain = np.zeros_like(flat)

        # First pixel
        plain[0] = (flat[0] - chaos[0]) % 256

        # Reverse diffusion
        for i in range(1, N):

            key = (
                chaos[i]
                + flat[i - 1]
                + ((i * 31) % 256)
            ) % 256

            plain[i] = (
                flat[i]
                - key
            ) % 256

        return plain.astype(np.uint8).reshape(image.shape)