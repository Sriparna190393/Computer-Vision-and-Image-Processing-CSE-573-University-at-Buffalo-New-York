"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.

    inlier_points_name = []
    outlier_points_name = []
    considered_point_pair = []
    #MAX = 0
    average_error = ''
    for iter in range(k):
        point1 = random.choice(input_points)
        point2 = random.choice(input_points)
        if (point1['name'], point2['name']) in considered_point_pair \
                and (point2['name'], point1['name']) in considered_point_pair:
            continue
        elif point1['name'] == point2['name'] or point1['value'][0] == point2['value'][0]:
            continue
        elif (point1['name'], point2['name']) not in considered_point_pair and (point2['name'], point1['name']) \
                not in considered_point_pair and point1['name'] != point2['name']:
            y_diff = point1['value'][1] - point2['value'][1]
            x_diff = point2['value'][0] - point1['value'][0]
            c = point1['value'][0]*point2['value'][1] - point2['value'][0]*point1['value'][1]
            x_coeff = y_diff
            y_coeff = x_diff
            inlier_temp_name = []
            outlier_temp_name = []
            sum_of_error = 0

            for item in input_points:
                if item['name'] != point1['name'] and item['name'] != point2['name']:
                    perp_distance = (abs((x_coeff * item['value'][0]) + (y_coeff * item['value'][1]) + c) / (
                                (x_coeff ** 2 + y_coeff ** 2) ** 0.5))
                    if perp_distance <= t:
                        inlier_temp_name.append(item['name'])
                        sum_of_error = sum_of_error + perp_distance
                    else:
                        outlier_temp_name.append(item['name'])
            if len(inlier_temp_name) >= d:
                curr_average_error = sum_of_error/len(inlier_temp_name)
                if average_error == '':
                    inlier_points_name = inlier_temp_name
                    inlier_points_name.append(point1['name'])
                    inlier_points_name.append(point2['name'])
                    outlier_points_name = outlier_temp_name
                    average_error = str(curr_average_error)
                elif curr_average_error < float(average_error):
                    inlier_points_name = inlier_temp_name
                    inlier_points_name.append(point1['name'])
                    inlier_points_name.append(point2['name'])
                    outlier_points_name = outlier_temp_name
                    average_error = str(curr_average_error)
                elif curr_average_error == float(average_error):
                    if len(inlier_temp_name) >= (len(inlier_points_name)-2):
                        inlier_points_name = inlier_temp_name
                        outlier_points_name = outlier_temp_name
                        inlier_points_name.append(point1['name'])
                        inlier_points_name.append(point2['name'])
                        average_error = str(curr_average_error)
            considered_point_pair.append((point1['name'], point2['name']))
    # raise NotImplementedError
    return inlier_points_name, outlier_points_name


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    # print(inlier_points_name,outlier_points_name)
    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
