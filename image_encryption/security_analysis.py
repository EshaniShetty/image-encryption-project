import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Histogram Analysis
# ============================================================

def histogram_analysis(original, encrypted):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.title("Original Histogram")

    for i,color in enumerate(("b","g","r")):
        hist=cv2.calcHist([original],[i],None,[256],[0,256])
        plt.plot(hist,color=color)

    plt.subplot(1,2,2)
    plt.title("Encrypted Histogram")

    for i,color in enumerate(("b","g","r")):
        hist=cv2.calcHist([encrypted],[i],None,[256],[0,256])
        plt.plot(hist,color=color)

    plt.show()


# ============================================================
# Grayscale Entropy
# ============================================================

def entropy_analysis(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hist=cv2.calcHist([gray],[0],None,[256],[0,256])

    hist=hist/hist.sum()

    entropy=0

    for p in hist:
        if p>0:
            entropy-=p*np.log2(p)

    return float(entropy)


# ============================================================
# RGB Entropy
# ============================================================

def rgb_entropy(image):

    values=[]

    for channel in range(3):

        hist=cv2.calcHist([image],[channel],None,[256],[0,256])

        hist=hist/hist.sum()

        entropy=0

        for p in hist:
            if p>0:
                entropy-=p*np.log2(p)

        values.append(float(entropy))

    return values


# ============================================================
# Horizontal Correlation
# ============================================================

def correlation_analysis(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    x=gray[:,:-1].flatten()

    y=gray[:,1:].flatten()

    return np.corrcoef(x,y)[0,1]


# ============================================================
# Horizontal Vertical Diagonal Correlation
# ============================================================

def correlation_all(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Horizontal

    x=gray[:,:-1].flatten()

    y=gray[:,1:].flatten()

    horizontal=np.corrcoef(x,y)[0,1]

    # Vertical

    x=gray[:-1,:].flatten()

    y=gray[1:,:].flatten()

    vertical=np.corrcoef(x,y)[0,1]

    # Diagonal

    x=gray[:-1,:-1].flatten()

    y=gray[1:,1:].flatten()

    diagonal=np.corrcoef(x,y)[0,1]

    return horizontal,vertical,diagonal


# ============================================================
# NPCR
# ============================================================

def npcr(img1,img2):

    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    return np.sum(gray1!=gray2)/gray1.size*100


# ============================================================
# UACI
# ============================================================

def uaci(img1,img2):

    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    return (
        np.mean(
            np.abs(
                gray1.astype(np.int16)-gray2.astype(np.int16)
            )
        )/255
    )*100


# ============================================================
# Differential Attack Analysis
# ============================================================

def differential_analysis(cipher1,cipher2):

    gray1=cv2.cvtColor(cipher1,cv2.COLOR_BGR2GRAY)

    gray2=cv2.cvtColor(cipher2,cv2.COLOR_BGR2GRAY)

    npcr_value=np.sum(gray1!=gray2)/gray1.size*100

    uaci_value=(
        np.mean(
            np.abs(
                gray1.astype(np.int16)-gray2.astype(np.int16)
            )
        )/255
    )*100

    return npcr_value,uaci_value


# ============================================================
# Local Shannon Entropy
# ============================================================

def local_entropy(image,
                  block_size=44,
                  samples=30):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    h,w=gray.shape

    entropies=[]

    for _ in range(samples):

        x=np.random.randint(0,h-block_size)

        y=np.random.randint(0,w-block_size)

        block=gray[x:x+block_size,
                   y:y+block_size]

        hist=cv2.calcHist([block],[0],None,[256],[0,256])

        hist=hist/hist.sum()

        entropy=0

        for p in hist:

            if p>0:
                entropy-=p*np.log2(p)

        entropies.append(entropy)

    return float(np.mean(entropies))


# ============================================================
# Chi Square
# ============================================================

def chi_square(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hist=cv2.calcHist([gray],[0],None,[256],[0,256])

    expected=gray.size/256

    chi=np.sum(
        ((hist-expected)**2)/expected
    )

    return float(chi)


# ============================================================
# MSE
# ============================================================

def mse(img1,img2):

    return np.mean(
        (img1.astype(float)-img2.astype(float))**2
    )


# ============================================================
# PSNR
# ============================================================

def psnr(img1,img2):

    error=mse(img1,img2)

    if error==0:
        return float("inf")

    return 20*np.log10(255/np.sqrt(error))


# ============================================================
# Key Sensitivity
# ============================================================

def key_difference(cipher1,cipher2):

    changed=np.sum(cipher1!=cipher2)

    total=cipher1.size

    return changed/total*100

# ---------------- Differential Attack ----------------

def differential_analysis(cipher1, cipher2):

    gray1 = cv2.cvtColor(cipher1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cipher2, cv2.COLOR_BGR2GRAY)

    npcr_value = (
        np.sum(gray1 != gray2) / gray1.size
    ) * 100

    uaci_value = (
        np.mean(
            np.abs(
                gray1.astype(np.int16) -
                gray2.astype(np.int16)
            )
        ) / 255
    ) * 100

    return npcr_value, uaci_value