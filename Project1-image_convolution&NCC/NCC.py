"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""

import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args


def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    # raise NotImplementedError
    n_temp = len(template) * len(template[0])  # number of elements in template
    n_patch = len(patch) * len(patch[0])  # number of elements in patch
    ncc_numerator = 0
    template_square_sum = 0
    patch_square_sum = 0
    len_temp = len(template)
    sum_template = sum(sum(template, []))
    temp_mean = sum_template / n_temp
    sum_patch = sum(sum(patch, []))
    patch_mean = sum_patch / n_patch
    for i in range(0, len_temp):
        for j in range(0, len(template[0])):
            ncc_numerator += (template[i][j] - temp_mean) * (patch[i][j] - patch_mean)
            template_square_sum += (template[i][j] - temp_mean) ** 2
            patch_square_sum += (patch[i][j] - patch_mean) ** 2
    ncc_denominator = (template_square_sum * patch_square_sum) ** 0.5
    ncc_value = ncc_numerator / ncc_denominator
    return ncc_value


def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    # raise NotImplementedError
    # raise NotImplementedError
    x_temp_len = len(template)  # rows : 24
    y_temp_len = len(template[0])  # columns : 26
    max_value = 0
    xmax = x_temp_len
    xmin = 0
    while xmax < len(img):
        ymin = 0
        ymax = y_temp_len
        while ymax < len(img[0]):
            patch = utils.crop(img, xmin, xmax, ymin, ymax)
            ncc_val = norm_xcorr2d(patch, template)
            if ncc_val > max_value:
                max_value = ncc_val
                x = xmin
                y = ymin
            ymin = ymin + 1
            ymax = ymax + 1
        xmin = xmin + 1
        xmax = xmax + 1
    return x, y, max_value


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)
    # match(img, template)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
