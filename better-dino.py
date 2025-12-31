import time
import numpy as np
import cv2
import mss
import pyautogui

DINO_X = 580
DINO_Y = 240
DINO_WIDTH = 600
DINO_HEIGHT = 155

# --- Trigger ROI (ahead of dino) ---
DETECT_X_REL = 43
STRIP_W = 110
STRIP_H_OFFSET = 60
STRIP_H = 50

# --- Added: Danger ROI (near dino front) ---
# This is the "don't descend into obstacle" safety check.
DANGER_X_REL = 20  # closer to dino
DANGER_W = 60
DANGER_Y_OFFSET = STRIP_H_OFFSET
DANGER_H = STRIP_H

# --- Added: Lookahead ROI (optional) ---
# Helps decide if there’s another obstacle soon, so dropping is useful.
LOOK_X_REL = 200
LOOK_W = 120
LOOK_Y_OFFSET = STRIP_H_OFFSET
LOOK_H = STRIP_H

WINDOW_NAME = "game_render"
JUMP_KEY = "space"

# detection threshold for "black-ish"
DARK_THR = 100

# timings
MIN_AIR_TIME = 0.17  # don't drop immediately after jump
DROP_HOLD = 0.03  # how long to hold DOWN to force descent


def roi_hit(frame, x_rel, w, y_off, h, thr=DARK_THR):
    H, W = frame.shape[:2]
    x1 = int(np.clip(x_rel, 0, W - 1))
    x2 = int(np.clip(x_rel + w, 0, W))
    y1 = int(np.clip(y_off, 0, H - 1))
    y2 = int(np.clip(y_off + h, 0, H))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, (x1, y1, x2, y2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return bool(np.any(gray < thr)), (x1, y1, x2, y2)


def jump():
    pyautogui.press(JUMP_KEY)


def main():
    print("Starting in 1 second... click the Chrome dino tab now.")
    time.sleep(1)

    game_monitor = {
        "left": DINO_X,
        "top": DINO_Y,
        "width": DINO_WIDTH,
        "height": DINO_HEIGHT,
    }

    last_jump_t = -999.0
    last_down_t = -999.0
    down_is_held = False

    with mss.mss() as sct:
        while True:
            game_img = np.array(sct.grab(game_monitor))  # BGRA
            game_frame = game_img[:, :, :3].copy()  # BGR

            now = time.time()

            trigger_hit, trigger_rect = roi_hit(
                game_frame, DETECT_X_REL, STRIP_W, STRIP_H_OFFSET, STRIP_H
            )
            danger_hit, danger_rect = roi_hit(
                game_frame, DANGER_X_REL, DANGER_W, DANGER_Y_OFFSET, DANGER_H
            )
            look_hit, look_rect = roi_hit(
                game_frame, LOOK_X_REL, LOOK_W, LOOK_Y_OFFSET, LOOK_H
            )

            # --- Jump logic ---
            if trigger_hit:
                jump()

            # --- Drop logic (hold DOWN briefly) ---
            # Only consider dropping after minimum airtime AND when danger zone is clear.
            should_drop = (
                (now - last_jump_t) > MIN_AIR_TIME
                and not danger_hit
                and look_hit  # only bother if another obstacle is coming soon
            )

            if should_drop and not down_is_held and (now - last_down_t) > (DROP_HOLD):
                pyautogui.keyDown("down")
                time.sleep(DROP_HOLD)
                pyautogui.keyUp("down")

            # --- overlays ---
            cv2.rectangle(
                game_frame,
                (trigger_rect[0], trigger_rect[1]),
                (trigger_rect[2] - 1, trigger_rect[3] - 1),
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                game_frame,
                (danger_rect[0], danger_rect[1]),
                (danger_rect[2] - 1, danger_rect[3] - 1),
                (0, 255, 255),
                2,
            )
            cv2.rectangle(
                game_frame,
                (look_rect[0], look_rect[1]),
                (look_rect[2] - 1, look_rect[3] - 1),
                (255, 0, 0),
                2,
            )

            cv2.putText(
                game_frame,
                f"trigger={trigger_hit} danger={danger_hit} look={look_hit} down={down_is_held}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, game_frame)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if down_is_held:
        pyautogui.keyUp("down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
