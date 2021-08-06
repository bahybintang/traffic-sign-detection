import cv2
import numpy as np
import imutils
import math
import functools
import matplotlib.pyplot as plt
import os


def identify_red(imag):
    img = imag.copy()

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # mask to extract red
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    red_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # separating channels
    r_channel = red_mask[:, :, 2]
    g_channel = red_mask[:, :, 1]
    b_channel = red_mask[:, :, 0]

    # filtering
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    filtered_r = 6 * filtered_r - 0.5 * filtered_b - 2 * filtered_g

    blank = np.uint8(filtered_r)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)
    _, r_thresh = cv2.threshold(opening, 10, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

    # if not cnts == []:
    #     cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    #     # c = cnts_sorted[0]
    #     # x, y, w, h = cv2.boundingRect(c)
    #     # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
    #     return cnts_sorted
    # else:
    #     return None


def identify_blue(imag):
    img = imag.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert the image to HSV format for color segmentation
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

    # mask to extract blue
    lower_blue = np.array([96, 127, 20])
    upper_blue = np.array([126, 255, 200])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # seperate out the channels
    r_channel = blue_mask[:, :, 2]
    g_channel = blue_mask[:, :, 1]
    b_channel = blue_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a blue gray space
    filtered_b = -0.5 * filtered_r + 4 * filtered_b - 2.5 * filtered_g

    blank = np.uint8(filtered_b)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, b_thresh = cv2.threshold(opening, 60, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

    # if not cnts == []:
    #     cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    #     # c = cnts_sorted[0]
    #     # x, y, w, h = cv2.boundingRect(c)
    #     # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
    #     return cnts_sorted
    # else:
    #     return None


def identify_yellow(imag):
    img = imag.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert the image to HSV format for color segmentation
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

    # mask to extract blue
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.bitwise_or(img_hsv, img_hsv, mask=mask)

    # seperate out the channels
    r_channel = yellow_mask[:, :, 2]
    g_channel = yellow_mask[:, :, 1]
    b_channel = yellow_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a yellow gray space
    filtered_b = 3 * filtered_r - 0.5 * filtered_b + 5 * filtered_g

    blank = np.uint8(filtered_b)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, b_thresh = cv2.threshold(opening, 50, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

    # if not cnts == []:
    #     cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    #     c = cnts_sorted[0]
    #     # x, y, w, h = cv2.boundingRect(c)
    #     # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
    #     return cnts_sorted
    # else:
    #     return None


def contourComparator(c1, c2):
    a1 = cv2.contourArea(c1)
    a2 = cv2.contourArea(c2)
    _, _, w1, _ = cv2.boundingRect(c1)
    _, _, w2, _ = cv2.boundingRect(c2)
    r1 = w1 / 2
    r2 = w2 / 2
    approx_area_1 = math.pi * r1 * r1
    approx_area_2 = math.pi * r2 * r2
    ratio_1 = a1 / approx_area_1
    ratio_2 = a2 / approx_area_2

    min_ratio_threshold = 0.9
    max_ratio_threshold = 1.1

    is_in_ratio_1 = ratio_1 >= min_ratio_threshold and ratio_1 <= max_ratio_threshold
    is_in_ratio_2 = ratio_2 >= min_ratio_threshold and ratio_2 <= max_ratio_threshold

    if a1 / a2 >= min_ratio_threshold and a1 / a2 <= max_ratio_threshold:
        # Prioritize circle if detected to eliminate noise
        if is_in_ratio_1 and not is_in_ratio_2:
            return 1
        elif is_in_ratio_2 and not is_in_ratio_1:
            return -1
        else:
            return 0
    else:
        return a1 - a2


def detect_sign(img, is_yolo=False):
    img_height, img_width, _ = img.shape

    red_list = identify_red(img)
    blue_list = identify_blue(img)
    yellow_list = identify_yellow(img)

    candidates = red_list + blue_list + yellow_list

    # candidates = sorted(
    #     candidates, key=functools.cmp_to_key(contourComparator), reverse=True)

    candidates = sorted(
        candidates, key=cv2.contourArea, reverse=True)

    # Check if aspect ratio match
    min_aspect_ratio = 0.75
    max_aspect_ratio = 1.5

    for c in candidates:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h
        if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
            if is_yolo:
                return (x + w/2.0) / img_width, (y + h/2.0) / img_height, w / img_width, h / img_height
            else:
                return x, y, w, h

    return None, None, None, None


def detect_sign_by_path(path, is_yolo=False):
    return detect_sign(cv2.imread(path), is_yolo)

# while True:
#     img = cv2.imread('test_images/blue_1.jpg')

#     red_list = identify_red(img)
#     blue_list = identify_blue(img)
#     yellow_list = identify_yellow(img)

#     candidates = red_list + blue_list + yellow_list

#     # candidates = sorted(
#     #     candidates, key=functools.cmp_to_key(contourComparator), reverse=True)

#     candidates = sorted(
#         candidates, key=cv2.contourArea, reverse=True)

#     found = False

#     # Check if aspect ratio match
#     min_aspect_ratio = 0.9
#     max_aspect_ratio = 1.1

#     for c in candidates:
#         x, y, w, h = cv2.boundingRect(c)
#         aspect_ratio = w / h

#         if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
#             cv2.rectangle(img, (x, y), (int(x + w),
#                                         int(y + h)), (0, 255, 0), 2)
#             found = True
#             break

#     # x, y, w, h = cv2.boundingRect(candidates[0])
#     # r = w / 2
#     # a = cv2.contourArea(candidates[0])

#     # # cv2.drawContours(img, [candidates[2]], 0,
#     # #                  (0, 255, 0), thickness=cv2.FILLED)

#     # aa = math.pi * r * r
#     # aspect_ratio = w / h
#     # print(aspect_ratio, w, h, a, aa, w * h)
#     # cv2.rectangle(img, (x, y), (int(x + w),
#     #                             int(y + h)), (0, 255, 0), 2)

#     if not found:
#         print("No traffic sign found!")

#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
