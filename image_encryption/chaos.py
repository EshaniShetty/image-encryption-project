import numpy as np

class SkewTentMap:

    def __init__(self, x0=0.45, p=0.35):

        if not (0 < x0 < 1):
            raise ValueError("x0 must be between 0 and 1")

        if not (0 < p < 1):
            raise ValueError("p must be between 0 and 1")

        self.x = np.float64(x0)
        self.p = np.float64(p)

    def generate_sequence(self, length, discard=1000):

        x = self.x
        p = self.p

        # Discard transient values
        for _ in range(discard):
            if x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)

        sequence = np.empty(length, dtype=np.float64)

        # Generate chaotic sequence
        for i in range(length):
            if x < p:
                x = x / p
            else:
                x = (1.0 - x) / (1.0 - p)

            sequence[i] = x

        self.x = x

        return sequence