"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    MIN_MATCH_COUNT = 10
    left_img = cv2.resize(left_img, (0, 0), fx=1, fy=1)
    right_img = cv2.resize(right_img, (0, 0), fx=1, fy=1)
    gray_left_image = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right_image = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    # create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_right_image, None)
    kp2, des2 = sift.detectAndCompute(gray_left_image, None)
    right = cv2.drawKeypoints(right_img, kp1, None)
    # cv2.imwrite("results/right_keypoints",right)
    left = cv2.drawKeypoints(left_img, kp2, None)
    # cv2.imwrite("results/left_keypoints",left)

    # matches features from left and right images
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good_match_list = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_match_list.append(m)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)
    img3 = cv2.drawMatches(right_img, kp1, left_img, kp2, good_match_list, None, **draw_params)
    # cv2.imwrite("results/matched_image",img3)

    # find homography
    if len(good_match_list) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match_list]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match_list]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = gray_right_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        gray_left_image = cv2.polylines(gray_left_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_match_list), MIN_MATCH_COUNT))
        matchesMask = None

    # imagestitching
    dst = cv2.warpPerspective(right_img, M, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    dst[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    # crop the image
    # crop top
    if not np.sum(dst[0]):
        dst = dst[1:]
    # crop bottom
    elif not np.sum(dst[-1]):
        dst = dst[:-2]
    # crop left
    elif not np.sum(dst[:, 0]):
        dst = dst[:, 1:]
    # crop right
    elif not np.sum(dst[:, -1]):
        dst = dst[:, :-2]
    return dst
    # raise NotImplementedError


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg', result_image)
