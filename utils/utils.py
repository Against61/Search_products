import cv2

def read_image(image_path: str):
    image = cv2.imread(str(image_path), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image