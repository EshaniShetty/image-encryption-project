from image_confusion import (
    load_image,
    display_image,
    save_image,
    print_image_info
)
from performance_analysis import(
    start_timer,stop_timer,print_execution_time
)
from security_analysis import (
    histogram_analysis,
    entropy_analysis,
    correlation_analysis,
    npcr,
    uaci,
    mse,
    psnr,rgb_entropy,chi_square,correlation_all,local_entropy
)
import matplotlib.pyplot as plt
import cv2
from chaos import SkewTentMap
from mpht_shuffle import MPHTShuffle
from substitution import DynamicSBox
from diffusion_v2 import ChaosDiffusionV2

import hashlib
import numpy as np

print("Loading Image...")

image = load_image("../sample_images/mountain.png")

print_image_info(image)

height, width, channels = image.shape
rounds = 2
pixels = height * width
rgb_values = pixels * channels

per_round = (
    height          # row permutation
    + width         # column permutation
    + pixels        # pixel permutation
    + 256           # S-box
    + rgb_values    # diffusion
)

total_values = rounds * per_round

print("\nGenerating Chaos Sequence...")

secret_key = "flower125"

# ---------- Plaintext-dependent hash ----------
image_hash = hashlib.sha256(
    image.tobytes()
).hexdigest()

combined_key = secret_key + image_hash

hash_value = hashlib.sha256(
    combined_key.encode()
).hexdigest()

hash_parts = [
    int(hash_value[i:i+8], 16)
    for i in range(0, 64, 8)
]

x0 = (hash_parts[0] % 1000000) / 1000000
if x0 == 0:
    x0 = 0.5

p = 0.1 + ((hash_parts[1] % 800000) / 1000000)

discard_iterations = 1000 + (hash_parts[2] % 4000)

print("Generated x0 :", x0)
print("Generated p  :", p)
print("Discard      :", discard_iterations)

chaos = SkewTentMap(
    x0=x0,
    p=p
)

sequence = chaos.generate_sequence(
    total_values,
    discard_iterations
)

print("Sequence Generated.")

print("\nGenerating Dynamic S-Box...")

dynamic_sbox = DynamicSBox(sequence)

print("Dynamic S-Box Generated.")

print("First 20 S-Box values:")
print(dynamic_sbox.sbox[:20])

mpht = MPHTShuffle(sequence)

diffusion = ChaosDiffusionV2(sequence)

enc_start=start_timer()
encrypted_image = image.copy()
permutations = []

N = height * width * channels

for r in range(rounds):

    print(f"\n========== Round {r+1} ==========")

    start = r * per_round

    round_sequence = sequence[start:start + per_round]

    mpht = MPHTShuffle(round_sequence)

    dynamic_sbox = DynamicSBox(round_sequence)

    diffusion = ChaosDiffusionV2(round_sequence)

    shuffled_image, permutation = mpht.shuffle(encrypted_image)
    print("First 20 permutation indices:")
    print(permutation[:20])

    print("Original first pixel :", encrypted_image.reshape(-1,3)[0])
    print("Shuffled first pixel :", shuffled_image.reshape(-1,3)[0])
    orig = encrypted_image.reshape(-1, 3)
    shuf = shuffled_image.reshape(-1, 3)

    changed = np.sum(np.any(orig != shuf, axis=1))
    print("Pixels moved:", changed, "/", len(orig))
    shuffled_display=shuffled_image.copy()
    permutations.append((permutation, round_sequence))

    encrypted_image = dynamic_sbox.substitute_image(shuffled_image)

    encrypted_image = diffusion.diffuse(encrypted_image)

    print("Diffusion Completed.")

encryption_time=stop_timer(enc_start)


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(shuffled_display, cv2.COLOR_BGR2RGB))
plt.title("Shuffled")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB))
plt.title("Encrypted")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\nChecking Pixels")

print("Original  :", image[0,0])
print("Encrypted :", encrypted_image[0,0])

save_image(
    "encrypted/encrypted_flower.png",
    encrypted_image
)

print("\nEncrypted Image Saved Successfully.")

# Decryption

dec_start=start_timer()
decrypted_image = encrypted_image.copy()

for r in range(rounds-1, -1, -1):

    permutation, round_sequence = permutations[r]

    mpht = MPHTShuffle(round_sequence)

    dynamic_sbox = DynamicSBox(round_sequence)

    diffusion = ChaosDiffusionV2(round_sequence)

    decrypted_image = diffusion.inverse_diffuse(decrypted_image)

    decrypted_image = dynamic_sbox.inverse_substitute(decrypted_image)

    decrypted_image = mpht.unshuffle(decrypted_image, permutation)
decryption_time=stop_timer(dec_start)

print(
    "Decryption Successful:",
    np.array_equal(
        image,
        decrypted_image
    )
)
print_execution_time(
    encryption_time,
    decryption_time
)
print("\n========== Security Analysis ==========")

# Histogram
histogram_analysis(
    image,
    encrypted_image
)

print("Original Entropy:", entropy_analysis(image))
print("Encrypted Entropy:", entropy_analysis(encrypted_image))
# Correlation
corr = correlation_analysis(
    encrypted_image
)
print(f"Correlation           : {corr:.6f}")

# NPCR
npcr_value = npcr(
    image,
    encrypted_image
)
print(f"NPCR                  : {npcr_value:.4f}%")

# UACI
uaci_value = uaci(
    image,
    encrypted_image
)
print(f"UACI                  : {uaci_value:.4f}%")

# MSE (Original vs Decrypted)
mse_value = mse(
    image,
    decrypted_image
)
print(f"MSE                   : {mse_value:.6f}")

# PSNR (Original vs Decrypted)
psnr_value = psnr(
    image,
    decrypted_image
)
print(f"PSNR                  : {psnr_value}")
print("\n========== Additional Security Analysis ==========")

# RGB Entropy
r, g, b = rgb_entropy(encrypted_image)
print(f"Red Entropy    : {r:.6f}")
print(f"Green Entropy  : {g:.6f}")
print(f"Blue Entropy   : {b:.6f}")

# Correlation
h, v, d = correlation_all(encrypted_image)
print(f"Horizontal Corr : {h:.6f}")
print(f"Vertical Corr   : {v:.6f}")
print(f"Diagonal Corr   : {d:.6f}")

# Local Entropy
print(f"Local Entropy   : {local_entropy(encrypted_image):.6f}")

# Chi Square
print(f"Chi Square      : {chi_square(encrypted_image):.6f}")