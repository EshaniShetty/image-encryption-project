import numpy as np

class DynamicSBox:
    def __init__(self, chaos_sequence):
        self.chaos_sequence = chaos_sequence
        self.sbox = self.generate_sbox()
        self.inverse_sbox = self.generate_inverse_sbox()

    def generate_sbox(self):
        """
        Generate a key-dependent dynamic S-Box using
        chaos-driven Fisher-Yates shuffle.
        """

        if len(self.chaos_sequence) < 512:
            raise ValueError("Chaos sequence must contain at least 512 values.")

        sbox = np.arange(256, dtype=np.uint8)

        chaos = np.array(self.chaos_sequence[:512])

        k = 0

        for i in range(255, 0, -1):
            j = int(chaos[k] * (i + 1))
            sbox[i], sbox[j] = sbox[j], sbox[i]
            k += 1

        return sbox

    def generate_inverse_sbox(self):
        """
        Generate inverse S-Box for decryption.
        """
        inverse = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            inverse[self.sbox[i]] = i

        return inverse

    def substitute_image(self, image):
        """
        Apply S-Box substitution to the image.
        """
        return self.sbox[image]

    def inverse_substitute(self, image):
        """
        Reverse the substitution during decryption.
        """
        return self.inverse_sbox[image]