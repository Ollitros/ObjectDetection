import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from PIL import Image
from Main.YOLO import yolo_pretrained_model


def main():
    sess = K.get_session()
    image = Image.open("data/images/input/dog.jpg")
    image_size = image.size
    image = image.resize((416, 416))

    image.save("data/images/output/dog.JPG", dpi=(100, 100))
    out_scores, out_boxes, out_classes = yolo_pretrained_model(sess, "data/images/output/dog.jpg")
    image = Image.open("data/images/output/dog.jpg")
    image = image.resize((image_size[0], image_size[1]))
    image.save("data/images/output/dog.JPG", dpi=(100, 100))


main()