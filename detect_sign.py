import cv2
import numpy as np
import imutils

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
    lower_red_2 = np.array([170, 70, 60])
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

    filtered_r = 4 * filtered_r - 0.5 * filtered_b - 2 * filtered_g

    blank = np.uint8(filtered_r)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)
    _, r_thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        # c = cnts_sorted[0]
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
        return cnts_sorted
    else:
        return None


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
    lower_blue = np.array([94, 127, 20])
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
    filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g

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
    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        # c = cnts_sorted[0]
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
        return cnts_sorted
    else:
        return None

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
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    yellow_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # seperate out the channels
    r_channel = yellow_mask[:, :, 2]
    g_channel = yellow_mask[:, :, 1]
    b_channel = yellow_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a yellow gray space
    filtered_b = -0.5 * filtered_r + 3 * filtered_b + 3 * filtered_g

    blank = np.uint8(filtered_b)

    cv2.imshow("y", blank)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, b_thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        c = cnts_sorted[0]
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
        return cnts_sorted
    else:
        return None

while True:
    img = cv2.imread('test_images/red_8.jpeg')

    red_list = identify_red(img)
    blue_list = identify_blue(img)
    yellow_list = identify_yellow(img)

    biggest_area_candidates = []
    if red_list:
        biggest_area_candidates.append(red_list[0])
    if blue_list:
        biggest_area_candidates.append(blue_list[0])
    if yellow_list:
        biggest_area_candidates.append(yellow_list[0])

    biggest_area_candidates = sorted(biggest_area_candidates,
                          key=cv2.contourArea, reverse=True)

    if len(biggest_area_candidates) > 0:
        c = biggest_area_candidates[0]
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
    else:
        print("No traffic sign found!")

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
