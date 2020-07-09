#!/usr/bin/env pypy
import numpy as np
from scipy.signal import argrelextrema
import imageio
from datetime import datetime
import time

def image_open(block_size):
    """Функция открытия изображения, возвращающая 3-х мерный NumPy массив"""
    image = imageio.imread('screen2.bmp')
    return image


def print_result(results, image, block_size):
    """Функция сохранения изображения-результата"""
    grey = (128, 128, 128) #smooth block
    blue = (0, 32, 255) #text block
    yellow = (255, 224, 32) #graphics block
    pink = (255, 208, 160) #image block

    h, w, z = image.shape

    for y in range(0, h):
        for x in range(0, w):
            value = results[y//block_size, x//block_size]

            if value == -1:
                R, G, B = grey
            elif value == 0:
                R, G, B = blue
            elif value == 1:
                R, G, B = yellow
            else:
                R, G, B = pink

            image[y, x, 0] = R
            image[y, x, 1] = G
            image[y, x, 2] = B

    imageio.imwrite('result.png', image)


def to_ycbcr(array):
    """Функция вычисления яркостной компоненты (с учетом клиппирования)"""
    h, w, z = array.shape
    block = np.zeros(shape=(h, w))
    for i in range(0, h):
        for j in range(0, w):
            block[i, j] = int(0.299 * array[i,j,0] + 0.587 * array[i,j,1] + 0.144 * array[i,j,2])

            if block[i, j] > 255:
                block[i, j] = 255
            if block[i, j] < 0:
                block[i, j] = 0

    block = np.asarray(block).reshape(-1)
    return block


def modes_calculate(frequences, probabilities, A, th):
    """Вычисление мод (modes)"""
    frequences = np.concatenate((np.zeros(1), frequences, np.zeros(1)), axis=0)
    extremums = argrelextrema(frequences, np.greater)[0] - 1

    c_probability = np.zeros(extremums.size)

    for i in range(0, extremums.size):
        index = extremums[i]
        for j in range(index - A, index + A):

            if j < 0 or j > 255:
                continue
            else:
                c_probability[i] += probabilities[j]

    c_probability = c_probability[c_probability > th]

    return c_probability


def li_lei_classify(block, block_size):
    """Функция классификация блоков методом Li and Lei"""
    A = 2
    T = 0.5
    Th = 0.6
    th = 0.08

    block = to_ycbcr(block)

    frequency = np.zeros(shape=(256))

    for i in range(0, block.size):
        frequency[block[i].astype(int)] += 1

    probability = frequency / (block_size * block_size)

    modes = modes_calculate(frequency, probability, A, th)
    N = modes.size

    if N == 1 and modes[0] > Th:
        return -1
    elif N == 2 and sum(modes) > Th and abs(modes[0] - modes[1]) > T :
        return 0
    elif N <= 4 and sum(modes) > Th:
        return 1
    else:
        return 2


if __name__ == '__main__':
    start_time = datetime.now()

    block_size = 16

    array = image_open(block_size)
    h, w, z = array.shape

    result = np.zeros(shape=(h, w))

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = array[y:y + block_size, x:x + block_size, :]
            result[y//block_size, x//block_size] = li_lei_classify(block, block_size)

    print_result(result, array, block_size)

    print("Time -> ", datetime.now() - start_time)

