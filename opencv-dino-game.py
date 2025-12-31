import time
import numpy as np
import cv2
import mss
import pyautogui

DINO_X = 580
DINO_Y = 240
DINO_WIDTH = 600
DINO_HEIGHT = 155
DETECT_X_REL = 83

STRIP_W = 70
STRIP_H_OFFSET = 60
STRIP_H = 50
WINDOW_NAME = "game_render"

JUMP_KEY = "space"


def jump():
    pyautogui.press(JUMP_KEY)


def main():
    print("Starting in 2 seconds... click the Chrome dino tab now.")
    time.sleep(2)

    game_monitor = {
        "left": DINO_X,
        "top": DINO_Y,
        "width": DINO_WIDTH,
        "height": DINO_HEIGHT,
    }

    with mss.mss() as sct:
        while True:
            game_img = np.array(sct.grab(game_monitor))  # BGRA
            game_frame = game_img[:, :, :3].copy()  # BGR

            h, w = game_frame.shape[:2]

            # Strip rectangle in GAME-FRAME coordinates (already relative)
            x1 = int(np.clip(DETECT_X_REL, 0, w - 1))
            x2 = int(np.clip(DETECT_X_REL + STRIP_W, 0, w))
            y1 = int(np.clip(STRIP_H_OFFSET, 0.0, h - 1))
            y2 = int(np.clip(STRIP_H_OFFSET + STRIP_H, 0, h))

            # Crop strip from the SAME pixels you display (perfect alignment)
            strip_roi = game_frame[y1:y2, x1:x2]

            # Detect obstacle in the strip
            gray = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
            hit = np.any(gray < 100)
            cv2.imshow("just showing", gray)
            cv2.setWindowProperty("just showing", cv2.WND_PROP_TOPMOST, 1)

            if hit:
                jump()

            # Draw overlays: line + rectangle
            cv2.rectangle(
                game_frame, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 0, 255), 2
            )

            # Optional: show hit status
            cv2.putText(
                game_frame,
                f"HIT={hit}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, game_frame)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
