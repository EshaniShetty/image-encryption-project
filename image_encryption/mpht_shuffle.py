import numpy as np


class MPHTShuffle:

    def __init__(self, chaos_sequence):
        self.chaos_sequence = chaos_sequence

    def shuffle(self, image):

        h, w, c = image.shape

        offset = 0

        # ---------- Row Permutation ----------
        row_perm = np.argsort(
            self.chaos_sequence[offset:offset + h]
        )
        offset += h

        shuffled = image[row_perm, :, :]

        # ---------- Column Permutation ----------
        col_perm = np.argsort(
            self.chaos_sequence[offset:offset + w]
        )
        offset += w

        shuffled = shuffled[:, col_perm, :]

        # ---------- Pixel Permutation ----------
        pixels = shuffled.reshape(-1, c)

        total_pixels = len(pixels)

        pixel_perm = np.argsort(
            self.chaos_sequence[offset:offset + total_pixels]
        )

        pixels = pixels[pixel_perm]

        shuffled = pixels.reshape(h, w, c)

        permutation = (
            row_perm,
            col_perm,
            pixel_perm
        )

        return shuffled, permutation


    def unshuffle(self, image, permutation):

        row_perm, col_perm, pixel_perm = permutation

        h, w, c = image.shape

        # ---------- Undo Pixel Permutation ----------
        pixels = image.reshape(-1, c)

        original_pixels = np.zeros_like(pixels)

        original_pixels[pixel_perm] = pixels

        restored = original_pixels.reshape(h, w, c)

        # ---------- Undo Column Permutation ----------
        inverse_col = np.argsort(col_perm)

        restored = restored[:, inverse_col, :]

        # ---------- Undo Row Permutation ----------
        inverse_row = np.argsort(row_perm)

        restored = restored[inverse_row, :, :]

        return restored