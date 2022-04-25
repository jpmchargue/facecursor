import cv2
import dlib
from imutils import face_utils
import math
import time
import pyautogui
import sys


def faceMouseLocation(shape):
    # Return the average location of the eye points.
    sum_x = sum([x for x, y in shape])
    sum_y = sum([y for x, y in shape])
    return (sum_x / 12.0, sum_y / 12.0)

def faceMouseRatios(shape):
    # Return ratios between eye width and eye height.
    width_l = math.dist(shape[0], shape[3])
    width_r = math.dist(shape[6], shape[9])
    width_avg = (width_l + width_r) / 2.0

    l_open_avg = (math.dist(shape[1], shape[5]) + math.dist(shape[2], shape[4])) / 2.0
    r_open_avg = (math.dist(shape[7], shape[11]) + math.dist(shape[8], shape[10])) / 2.0

    l_ratio = width_avg / max(l_open_avg, 0.01)
    r_ratio = width_avg / max(r_open_avg, 0.01)

    return (l_ratio, r_ratio)

camera = cv2.VideoCapture(0)

# Face Cursor Logistical Variables
#rolling_ratio = (2, 2)
#rolling_factor = 0.85
screen_size = pyautogui.size()
click_threshold = 6

left_down = False
right_down = False

calibrating = False
min_x = 0
max_x = 0
min_y = 0
max_y = 0
mouse_smooth = (screen_size[0] / 2.0, screen_size[1] / 2.0)
mouse_smooth_factor = 0.90

left_counter = 0
right_counter = 0
wink_frames = 5

last_click = time.time()
cooldown = (1/3) # minimum of 1/3 seconds between clicks

hide = (len(sys.argv) == 2 and sys.argv[1] == "hide")

if camera.isOpened():
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("predictor.dat")

    print("*** CALIBRATION ***")
    print("Tilt your head up slightly, then right slightly, then down, etc.")
    print("rotating it in a slow, clockwise fashion.")
    print("The program will begin in 10 seconds.")
    calibration_start_time = time.time()
    while time.time() - calibration_start_time < 10:
        success, frame = camera.read()
        detected_faces = face_detector(frame, 0)
        if len(detected_faces) == 1:
            for d in detected_faces:
                shape = face_utils.shape_to_np(predictor(frame, d))
                center = faceMouseLocation(shape)

                if not calibrating:
                    min_x = center[0]
                    max_x = center[0]
                    min_y = center[1]
                    max_y = center[1]
                    calibrating = True
                else:
                    if center[0] < min_x:
                        min_x = center[0]
                    if center[0] > max_x:
                        max_x = center[0]
                    if center[1] < min_y:
                        min_y = center[1]
                    if center[1] > max_y:
                        max_y = center[1]


    #cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    print("*** STARTING ***")
    last_time = time.time()
    while True:
        success, frame = camera.read()
        if success:
            if True:
                detected_faces = face_detector(frame, 0)
                if len(detected_faces) == 1:
                    for d in detected_faces:
                        shape = face_utils.shape_to_np(predictor(frame, d))
                        for (x, y) in shape:
                            cv2.circle(frame, (round(x), round(y)), radius=2, color=(255, 255, 0), thickness=-1)

                        location = faceMouseLocation(shape)
                        ratios = faceMouseRatios(shape)
                        #print(ratios)
                        #rolling_ratio = ((rolling_ratio[0] * rolling_factor) + (ratios[0] * (1-rolling_factor)),
                        #    (rolling_ratio[1] * rolling_factor) + (ratios[1] * (1-rolling_factor)))

                        # Display current eye width ratios
                        cv2.putText(frame, str(round(ratios[0], 2)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                        cv2.putText(frame, str(round(ratios[1], 2)), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

                        # Display center of gaze
                        cv2.circle(frame, (round(location[0]), round(location[1])), radius=2, color=(0, 255, 0), thickness=-1)

                        # Display calibration edges
                        cv2.circle(frame, (round(min_x), round(min_y)), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.circle(frame, (round(max_x), round(min_y)), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.circle(frame, (round(max_x), round(max_y)), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.circle(frame, (round(min_x), round(max_y)), radius=2, color=(0, 0, 255), thickness=-1)

                        # Manage mouse movement
                        mouse_x = screen_size[0] * (max_x - location[0]) / (max_x - min_x)
                        mouse_y = screen_size[1] * (location[1] - min_y) / (max_y - min_y)
                        mouse_smooth = ((mouse_smooth_factor * mouse_smooth[0]) + ((1-mouse_smooth_factor) * mouse_x),
                            (mouse_smooth_factor * mouse_smooth[1]) + ((1-mouse_smooth_factor) * mouse_y))
                        pyautogui.moveTo(mouse_smooth[0], mouse_smooth[1])

                        # Manage clicking
                        now = time.time()
                        if not right_down:
                            if ratios[1] > click_threshold and not ratios[0] > click_threshold:
                                right_down = True
                                if now - last_click > cooldown:
                                    print("LEFT CLICK")
                                    pyautogui.click()
                                    last_click = now
                        else:
                            if ratios[1] < click_threshold:
                                right_down = False
                        if not left_down:
                            if ratios[0] > click_threshold and not ratios[1] > click_threshold:
                                left_down = True
                                if now - last_click > cooldown:
                                    print("RIGHT CLICK")
                                    pyautogui.click(button='right')
                                    last_click = now
                        else:
                            if ratios[0] < click_threshold:
                                left_down = False


                        if False:
                            # After watching the ratio data for a bunch of test winks,
                            # a consistent indicator for a wink seemed to be if one eye's ratio is both above a certain threshold
                            # AND consistently greater than the other eye's for at least 5 frames.
                            if now - last_click > cooldown:
                                if ratios[0] > click_threshold or ratios[1] > click_threshold:
                                    if ratios[0] > ratios[1]:
                                        left_counter += 1
                                        right_counter = 0
                                    elif ratios[1] > ratios[0]:
                                        right_counter += 1
                                        left_counter = 0
                                    if left_counter == wink_frames:
                                        print("RIGHT CLICK")
                                        last_click = now
                                    if right_counter == wink_frames:
                                        print("LEFT CLICK")
                                        last_click = now
                                else:
                                    left_counter = 0
                                    right_counter = 0



                        fps = 1 / (now - last_time)
                        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                        last_time = now

            if not hide:
                cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    print("Could not get access to camera, quitting")

camera.release()
cv2.destroyAllWindows()
