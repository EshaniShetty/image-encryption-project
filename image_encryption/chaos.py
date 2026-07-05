import numpy as np

class SkewTentMap:
    """
    Skew Tent Map for generating chaotic sequences.
    """

    def __init__(self, x0=0.45, p=0.35):
        if not (0 < x0 < 1):
            raise ValueError("x0 must be between 0 and 1")

        if not (0 < p < 1):
            raise ValueError("p must be between 0 and 1")

        self.x = x0
        self.p = p

    def next_value(self):
        """
        Generate the next chaotic value.
        """

        if self.x < self.p:
            self.x = self.x / self.p
        else:
            self.x = (1 - self.x) / (1 - self.p)

        return self.x

    def generate_sequence(self, length):
        """
        Generate a chaotic sequence.
        """

        sequence = np.zeros(length)

        for i in range(length):
            sequence[i] = self.next_value()

        return sequence