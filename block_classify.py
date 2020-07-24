#!/usr/bin/env pypy
import numpy as np
import imageio
from datetime import datetime
import time
from collections import Counter

def image_open(image_name):
    """Функция открытия изображения, возвращающая 3-х мерный NumPy массив"""
    image = imageio.imread(image_name)
    return image


def print_result(results, image, block_size, filename):
    """Функция сохранения изображения-результата"""
    blue = (0, 63, 208) #блок, распознанный как текстовый
    pink = (255, 105, 180) #блок, распознанный как изображение

    h, w, z = image.shape

    for y in range(h//block_size):
        for x in range(w//block_size):
            value = results[y, x]

            if value == -1:
                R, G, B = blue
            else:
                R, G, B = pink

            if block_size * y + 1 < h:
                y_start = block_size * y + 1
            else:
                break

            if block_size * y + block_size < h:
                y_stop = block_size * y + block_size
            else:
                y_stop = h-1

            if block_size * x + 1 < w:
                x_start = block_size * x + 1
            else:
                break

            if block_size * x + block_size < w:
                x_stop = block_size * x + block_size
            else:
                x_stop = w-1

            image[y_start: y_stop, x_start, 0] = R
            image[y_start: y_stop, x_start, 1] = G
            image[y_start: y_stop, x_start, 2] = B

            image[y_start: y_stop, x_stop, 0] = R
            image[y_start: y_stop, x_stop, 1] = G
            image[y_start: y_stop, x_stop, 2] = B

            image[y_start, x_start: x_stop, 0] = R
            image[y_start, x_start: x_stop, 1] = G
            image[y_start, x_start: x_stop, 2] = B

            image[y_stop, x_start: x_stop, 0] = R
            image[y_stop, x_start: x_stop, 1] = G
            image[y_stop, x_start: x_stop, 2] = B

    imageio.imwrite(filename, image)


def classify(block, block_size, th, filter_val):
    """Функция классификации блока"""
    colors = []

    h, w, z = block.shape

    for y in range(w):
        for x in range(h):
            color = block[x,y]
            color_str = "" + str(color[0]) + "_" + str(color[1]) + "_" + str(color[2])
            colors.append(color_str)

    counter = Counter(colors)

    if filter_val > 0:#фильтрация шумоподобных цветов
        new_counter = Counter([el for el in colors if counter[el] >= filter_val])
        colors_number = len(list(new_counter))
    else:
        colors_number = len(list(counter))

    if colors_number > th:
        return 2 #блок естественного изображения
    else:
        return -1 #текстовый блок


if __name__ == '__main__':

    block_size = 16 #размер блока
    th = 4 #пороговое значение
    filter_value = 3 #параметр фильтра

    #original_image = "block"
    #original_image = "white-test"
    #original_image = "text-test"
    original_image = "text-graph"
    #original_image = "bad-quality"
    #original_image = "mixed"
    #original_image = "infographic"
    #original_image = "colored-text"
    #original_image = "picture"
    #original_image = "bad"

    input_filename = "pictures/" + original_image + ".bmp"
    image = image_open(input_filename)

    if filter_value != 0:
        output_filename = "results/" + original_image + "/filtered/f" + str(filter_value) + "/th=" + str(th) + ".png"
    else:
        output_filename = "results/" + original_image + "/" + "th=" + str(th) + ".bmp"

    image = image_open(input_filename)

    result = np.zeros(shape=(image.shape[0], image.shape[1]))

    for y in range(0, image.shape[0], block_size):
        for x in range(0, image.shape[1], block_size):
            block = image[y:y + block_size, x:x + block_size, :]
            result[y//block_size, x//block_size] = classify(block, block_size, th, filter_value)

    print_result(result, image, block_size, output_filename)