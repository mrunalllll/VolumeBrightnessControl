import cv2
import mediapipe as mp
import numpy as np
import math
import screen_brightness_control as sbc

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# VOLUME SETUP
devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

# MEDIAPIPE SETUP
mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# CAMERA
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_no, handLms in enumerate(results.multi_hand_landmarks):

            handType = results.multi_handedness[hand_no].classification[0].label

            h, w, c = img.shape

            # Thumb Tip
            x1 = int(handLms.landmark[4].x * w)
            y1 = int(handLms.landmark[4].y * h)

            # Index Tip
            x2 = int(handLms.landmark[8].x * w)
            y2 = int(handLms.landmark[8].y * h)

            length = math.hypot(x2 - x1, y2 - y1)

            percent = int(
                np.interp(length, [20, 150], [0, 100])
            )

            # RIGHT HAND = VOLUME
            if handType == "Right":

                vol = np.interp(
                    length,
                    [20, 150],
                    [volMin, volMax]
                )

                volume.SetMasterVolumeLevel(vol, None)

                cv2.putText(
                    img,
                    f"Volume: {percent}%",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # LEFT HAND = BRIGHTNESS
            elif handType == "Left":

                try:
                    sbc.set_brightness(percent)
                except:
                    pass

                cv2.putText(
                    img,
                    f"Brightness: {percent}%",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                3
            )

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("Volume & Brightness Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()