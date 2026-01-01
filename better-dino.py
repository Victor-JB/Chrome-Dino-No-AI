import time
import numpy as np
import cv2
import mss
import pyautogui
import argparse

p = argparse.ArgumentParser(description="Dino bot (width-adaptive, no queue)")
p.add_argument(
    "--invert", action="store_true", help="Invert grayscale before thresholding"
)
args = p.parse_args()

# ----------------- SCREEN / WINDOW -----------------
DINO_X = 580
DINO_Y = 240
DINO_WIDTH = 600
DINO_HEIGHT = 155
WINDOW_NAME = "game_render"

# ----------------- DETECTION -----------------
DARK_THR = 100  # "black-ish" threshold (after optional invert)

# Lookahead ROI: big enough to see full cactus clusters
LOOK_X_REL = 55
LOOK_W = 450
LOOK_Y_OFF = 70
LOOK_H = 50

# Base trigger x (where you'd jump for "normal" cactus)
SMALL_JUMP_X = 158
LARGE_JUMP_X = 140

# Width classification and trigger adjustment
LARGE_PX = 5

# Column hit heuristic: column counts as "occupied" if >= this fraction are obstacle pixels
COL_HIT_FRAC = 0.4
MIN_RUN = 2

# ----------------- INPUT / TIMING -----------------
JUMP_KEY = "space"

# Fast drop (DOWN) parameters
MIN_AIR_TIME = 0.09
DROP_HOLD = 0.04

# Safe x: only fast-drop once obstacle trailing edge is left of this x
SAFE_CLEAR_X = 100  # near dino front; tune (20-40)


def jump():
    pyautogui.press(JUMP_KEY)


def get_roi(game_frame, x_rel, w, y_off, h):
    H, W = game_frame.shape[:2]
    x1 = int(np.clip(x_rel, 0, W - 1))
    x2 = int(np.clip(x_rel + w, 0, W))
    y1 = int(np.clip(y_off, 0, H - 1))
    y2 = int(np.clip(y_off + h, 0, H))
    return game_frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def detect_next_obstacle_block(game_frame):
    """
    Returns None or dict:
      {"lead_x": int, "trail_x": int, "width_px": int, "rect": (x1,y1,x2,y2)}
    All x are in GAME-FRAME coordinates.
    """
    roi, rect = get_roi(game_frame, LOOK_X_REL, LOOK_W, LOOK_Y_OFF, LOOK_H)
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if args.invert:
        gray = 255 - gray

    mask = gray < DARK_THR
    col_frac = mask.mean(axis=0)
    col_hit = col_frac >= COL_HIT_FRAC

    idx = np.flatnonzero(col_hit)
    if idx.size == 0:
        return None

    lead = int(idx[0])
    trail = lead
    while trail + 1 < col_hit.shape[0] and col_hit[trail + 1]:
        trail += 1

    width_px = int(trail - lead + 1)
    if width_px < MIN_RUN:
        return None

    x1, y1, x2, y2 = rect
    return {
        "lead_x": x1 + lead,
        "trail_x": x1 + trail,
        "width_px": width_px,
        "rect": rect,
    }


def trigger_x_for_width(width_px):
    """
    Wider obstacle => jump earlier (smaller x).
    """
    if width_px > LARGE_PX:
        return LARGE_JUMP_X
    else:
        return SMALL_JUMP_X


def main():
    print("Starting in 1 second... click the Chrome dino tab now.")
    time.sleep(1)

    monitor = {
        "left": DINO_X,
        "top": DINO_Y,
        "width": DINO_WIDTH,
        "height": DINO_HEIGHT,
    }

    last_jump_t = -999.0
    last_drop_t = -999.0

    # State: avoid repeated jumps on the same obstacle
    armed = True  # can jump when armed
    in_air = False

    with mss.mss() as sct:
        while True:
            now = time.time()

            game_img = np.array(sct.grab(monitor))
            game_frame = game_img[:, :, :3].copy()
            h, w = game_frame.shape[:2]

            obs = detect_next_obstacle_block(game_frame)

            # Update in_air using time since jump (simple but effective)
            if in_air and (now - last_jump_t) > 0.1:
                # conservative "landed" time; tune if you want
                in_air = False

            # Re-arm once the current obstacle is clearly behind us
            if obs is not None:
                if obs["trail_x"] < SAFE_CLEAR_X:
                    armed = True
            else:
                # no obstacle visible => safe to arm
                armed = True

            # --- Jump decision ---
            if obs is not None:
                trig_x = trigger_x_for_width(obs["width_px"])
                print("triggered width: ", obs["width_px"])
                can_jump = armed

                if can_jump and obs["lead_x"] <= trig_x:
                    jump()
                    last_jump_t = now
                    in_air = True
                    armed = False  # disarm until this obstacle passes

            # --- Fast drop decision (only if airborne, only after obstacle cleared) ---
            if in_air and obs is not None:
                safe_to_drop = (now - last_jump_t) > MIN_AIR_TIME and (
                    now - last_drop_t
                ) > (DROP_HOLD + 0.02)
                if safe_to_drop:
                    pyautogui.keyDown("down")
                    time.sleep(DROP_HOLD)
                    pyautogui.keyUp("down")
                    last_drop_t = time.time()

            # --- Debug overlays ---
            # Lookahead rect
            look_roi, look_rect = get_roi(
                game_frame, LOOK_X_REL, LOOK_W, LOOK_Y_OFF, LOOK_H
            )
            cv2.rectangle(
                game_frame,
                (look_rect[0], look_rect[1]),
                (look_rect[2] - 1, look_rect[3] - 1),
                (255, 0, 0),
                2,
            )

            # Show trigger / obstacle edges
            if obs is not None:
                trig_x = trigger_x_for_width(obs["width_px"])
                cv2.line(
                    game_frame,
                    (obs["lead_x"], 0),
                    (obs["lead_x"], h - 1),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    game_frame,
                    (obs["trail_x"], 0),
                    (obs["trail_x"], h - 1),
                    (0, 255, 255),
                    2,
                )
                cv2.line(game_frame, (trig_x, 0), (trig_x, h - 1), (0, 0, 255), 2)
                cv2.putText(
                    game_frame,
                    f"w={obs['width_px']} trig={trig_x}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    game_frame,
                    "no obstacle",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                game_frame,
                f"armed={armed} in_air={in_air}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
