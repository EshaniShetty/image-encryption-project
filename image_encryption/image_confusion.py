import cv2
import matplotlib.pyplot as plt


def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    return image


def save_image(path, image):

    cv2.imwrite(path, image)


def display_image(title, image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def print_image_info(image):

    print("\nImage Information")
    print("----------------------")
    print("Height :", image.shape[0])
    print("Width  :", image.shape[1])
    print("Channel:", image.shape[2])
    print("Datatype:", image.dtype)

  