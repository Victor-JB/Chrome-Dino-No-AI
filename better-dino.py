import time
import numpy as np
import cv2
import mss
import pyautogui
import argparse

p = argparse.ArgumentParser(description="Config")
p.add_argument("--invert", action="store_true", help="Toggle if on dark mode")
args = p.parse_args()

DINO_X = 580.0
DINO_Y = 240
DINO_WIDTH = 600
DINO_HEIGHT = 155

# --- Trigger ROI (ahead of dino) ---
DETECT_X_REL = 83
STRIP_W = 33
STRIP_H_OFFSET = 80
STRIP_H = 30

# --- Added: Danger ROI (near dino front) ---
# This is the "don't descend into obstacle" safety check.
DANGER_X_REL = 20  # closer to dino
DANGER_W = 60
DANGER_Y_OFFSET = STRIP_H_OFFSET
DANGER_H = STRIP_H

# --- Added: Lookahead ROI (optional) ---
# Helps decide if there’s another obstacle soon, so dropping is useful.
LOOK_X_REL = DETECT_X_REL + STRIP_W
LOOK_W = 450
LOOK_Y_OFFSET = STRIP_H_OFFSET - 30
LOOK_H = STRIP_H + 30

WINDOW_NAME = "game_render"
JUMP_KEY = "space"

# detection threshold for "black-ish"
DARK_THR = 100

# timings
MIN_AIR_TIME = 0.09  # don't drop immediately after jump
DROP_HOLD = 0.04  # how long to hold DOWN to force descent

queue_list = []


def get_lookahead_roi(game_frame, x_rel, w, y_off, h):
    """
    Returns (roi_bgr, rect) where rect=(x1,y1,x2,y2) in GAME-FRAME coords.
    """
    H, W = game_frame.shape[:2]
    x1 = int(np.clip(x_rel, 0, W - 1))
    x2 = int(np.clip(x_rel + w, 0, W))
    y1 = int(np.clip(y_off, 0, H - 1))
    y2 = int(np.clip(y_off + h, 0, H))
    return game_frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# region of interest
def roi_hit(frame, x_rel, w, y_off, h, thr=DARK_THR):
    roi, coords = get_lookahead_roi(frame, x_rel, w, y_off, h)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray if args.invert else gray
    return bool(np.any(gray < thr)), coords


def find_next_obstacle(frame, thr=100, invert=False, col_hit_frac=0.08, min_run=2):
    roi_bgr, _ = get_lookahead_roi(frame, LOOK_X_REL, LOOK_W, LOOK_Y_OFFSET, LOOK_H)
    if roi_bgr.size == 0:
        return False, 0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = 255 - gray

    mask = gray < thr
    col_frac = mask.mean(axis=0)
    col_hit = col_frac >= col_hit_frac

    idx = np.flatnonzero(col_hit)
    if idx.size == 0:
        return False, 0

    lead = int(idx[0])
    trail = lead
    while trail + 1 < col_hit.shape[0] and col_hit[trail + 1]:
        trail += 1

    width_px = int(trail - lead + 1)
    if width_px < min_run:
        return False, 0

    queue_list.append(width_px)
    return True, width_px


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

            found, width_px = find_next_obstacle(game_frame)

            danger_hit, danger_rect = roi_hit(
                game_frame, DANGER_X_REL, DANGER_W, DANGER_Y_OFFSET, DANGER_H
            )

            # --- Jump logic ---
            if found:
                # jump when the obstacle reaches that trigger x
                # (add your cooldown here; example:)
                if lead_x_game <= trig_x and (now - last_jump_t) > 0.12:
                    jump()
                    last_jump_t = now

                    # draw where obstacle starts + where you will jump
                cv2.line(
                    game_frame,
                    (lead_x_game, 0),
                    (lead_x_game, LOOK_H - 1),
                    (0, 255, 0),
                    2,
                )  # green = obstacle lead
                cv2.line(
                    game_frame, (trig_x, 0), (trig_x, LOOK_H - 1), (0, 0, 255), 2
                )  # red = trigger
                cv2.putText(
                    game_frame,
                    f"w={width_px}px",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # --- Drop logic (hold DOWN briefly) ---
            # Only consider dropping after minimum airtime AND when danger zone is clear.
            should_drop = (now - last_jump_t) > MIN_AIR_TIME and not danger_hit

            if should_drop and not down_is_held and (now - last_down_t) > (DROP_HOLD):
                pyautogui.keyDown("down")
                time.sleep(DROP_HOLD)
                pyautogui.keyUp("down")

            # --- overlays ---
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
                f"trigger={found} danger={danger_hit} incoming={queue_list[0]} down={down_is_held}",
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
