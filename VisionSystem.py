import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys


def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):

    image = image.copy()

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if contour_sizes:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

    else:
        biggest_contour = image
        mask = [0, 0, 0]

    return biggest_contour, mask


def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    # easy function
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, blue, 2, cv2.FONT_HERSHEY_SIMPLEX)

    blue_positionX = (int)(x)
    blue_positionY = (int)(y)

    print("image shape: ", image.shape)
    print("blue robot position: (", blue_positionX, ", ", blue_positionY, ")")

    return image_with_ellipse, blue_positionX, blue_positionY


def find_red_cubes(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)
    scale = 700 / max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape, np.uint8)
    overlay = overlay_mask(mask_clean, image)
    image_with_ellipse = overlay.copy()
    cubes_num = 0

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 1000):
            cubes_num = cubes_num + 1
            cv2.drawContours(mask, [contour], -1, 255, -1)
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            cv2.ellipse(image_with_ellipse, ellipse, red, 2, cv2.FONT_HERSHEY_SIMPLEX)

            '''
            M = cv2.moments(contour)
            red_positionX[cubes_num-1] = int(M["m10"] / M["m00"])
            red_positionY[cubes_num-1] = int(M["m01"] / M["m00"])
            '''
            red_positionX[cubes_num - 1] = (int)(x)
            red_positionY[cubes_num - 1] = (int)(y)

    print("num of cubes found: ", cubes_num)
    for i in range(cubes_num):
        print("Position", i, "(", red_positionX[i], ", ", red_positionY[i], ")")

    # show(image_with_ellipse)

    bgr = cv2.cvtColor(image_with_ellipse, cv2.COLOR_RGB2BGR)

    return bgr


def find_blue_robot(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)
    scale = 700 / max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_blue = np.array([90, 63, 137])
    max_blue = np.array([108, 255, 255])

    mask = cv2.inRange(image_blur_hsv, min_blue, max_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_contour, mask_cont = find_biggest_contour(mask_clean)

    if (big_contour.any() != 0):
        overlay = overlay_mask(mask_clean, image)
        circled, blue_positionX, blue_positionY = circle_contour(overlay, big_contour)
        # show(circled)
        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        topX, topY, width, height = cv2.boundingRect(big_contour)
    else:
        bgr = image
        blue_positionX = None
        blue_positionY = None

    return bgr, blue_positionX, blue_positionY, topX, topY, width, height


def create_tracker():

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[1]

    
    tracker = cv2.Tracker_create(tracker_type)

    return tracker


def track():
    
    video = cv2.VideoCapture('http://admin:1234@192.168.43.127/mjpg/video.mjpg')
    #video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    frame_counter = -1

    while True:
        ok, frame = video.read()
        if not ok: break
        frame_counter += 1
        print("Number of frames: ", frame_counter)

        result2, blue_positionX, blue_positionY, topX, topY, width, height = find_blue_robot(frame)

        if (frame_counter % 10) == 0 :
            bounding_box = (topX, topY, width, height)
            tracker = create_tracker()
            ok = tracker.init(result2, bounding_box)
        

        # result1 = find_red_cubes(img)
        # cv2.imshow('result1', result1)

        
        # Start timer
        timer = cv2.getTickCount()

        ok, bounding_box = tracker.update(result2)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            # Tracking success
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
            cv2.rectangle(result2, p1, p2, (255, 255, 255), 1, 1)
        else:
            cv2.putText(result2, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),2)


        cv2.putText(result2, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", result2)


        k = cv2.waitKey(1) & 0xff
        if k == 27: break


def main():

    global cubes_num, red, green, blue, red_positionX, red_positionY
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    red_positionX = np.zeros((1, 15))
    red_positionY = np.zeros((1, 15))
    blue_positionX = 0
    blue_positionY = 0

    track()


if __name__ == '__main__':
    main()




