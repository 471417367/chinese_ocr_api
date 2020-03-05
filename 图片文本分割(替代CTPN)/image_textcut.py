import os
import cv2
import numpy as np
from IPython.display import display
from PIL import Image


def extract_peek(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            if i - start_i >= minimun_range:
                end_i = i
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


def img_text_cut(img_path, min_val=10000, min_range=10):
    '''
    :param img_path:
    :param min_val: # 每行像素和的阈值，超过10000的认为属于文本行
    :param min_range: # 当连续满足阈值的行超过10行，认为是文本行
    :return:
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)

    display(Image.fromarray(adaptive_threshold))

    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    print(horizontal_sum)

    peek_ranges = extract_peek(horizontal_sum, min_val, min_range)

    for y0, y1 in peek_ranges:
        print(y0, y1)
        display(Image.fromarray(img[y0:y1, ]))


img_text_cut('1.jpg')
