import time
import numpy as np
import cv2
import mss

# --------- CONFIG (EDIT THESE) ----------
DINO_LEFT = 580  # absolute screen x
DINO_TOP = 240  # reference y (ground-ish)
DINO_WIDTH = 600
DINO_HEIGHT = 155
DINO_DETECT_H = 40
DETECT_X_REL = 240  # relative to DINO_X

DELAY_SEC = 2.0
# ---------------------------------------

print("2 seconds to focus Chrome Dino window...")
time.sleep(DELAY_SEC)

with mss.mss() as sct:
    img = np.array(
        sct.grab(
            {
                "left": DINO_LEFT,
                "top": DINO_TOP,
                "width": DINO_WIDTH,
                "height": DINO_HEIGHT,
            }
        )
    )

frame = img[:, :, :3].copy()  # BGR

h, w = frame.shape[:2]

# Draw vertical red lines
cv2.line(frame, (DETECT_X_REL, 0), (DETECT_X_REL, h), (0, 0, 255), 2)

input("show? ")
cv2.imshow("Dino calibration", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
