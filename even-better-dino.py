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
SMALL_JUMP_X = 156
LARGE_JUMP_X = 160

# Width classification
LARGE_PX = 45

# ----------------- INPUT / TIMING -----------------
JUMP_KEY = "space"

# NEW: drop tuning (these work together)
MIN_AIR_TIME = 0.09  # don't attempt fast-drop immediately after jump
DROP_HOLD = 0.1  # how long to hold DOWN to accelerate descent

# NEW: "land just behind obstacle" safety + timing gate
SAFE_CLEAR_X = 100  # must be near dino front in GAME coords (tune ~20-120)
# NOTE: you currently use 100; keep for now since it works in your coordinate frame.

# For width detection robustness
OCC_THRESH = 0.12
GAP_PX = 4
MIN_RUN = 2


def jump():
    pyautogui.press(JUMP_KEY)


def fast_drop(hold_s: float):
    """
    Press/hold DOWN briefly to accelerate descent.
    (If you're already on the ground, this just ducks for a moment; we gate it with state.)
    """
    pyautogui.keyDown("down")
    time.sleep(hold_s)
    pyautogui.keyUp("down")


def get_roi(game_frame, x_rel, w, y_off, h):
    H, W = game_frame.shape[:2]
    x1 = int(np.clip(x_rel, 0, W - 1))
    x2 = int(np.clip(x_rel + w, 0, W))
    y1 = int(np.clip(y_off, 0, H - 1))
    y2 = int(np.clip(y_off + h, 0, H))
    return game_frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _column_occupancy_frac(roi_bgr: np.ndarray, thr: int, invert: bool) -> np.ndarray:
    """Fraction (0..1) of obstacle-like pixels per column."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = 255 - gray
    mask = gray < thr
    return mask.mean(axis=0)


def _close_1d(col_hit: np.ndarray, gap_px: int) -> np.ndarray:
    """Bridge small gaps in the 1D occupied-columns signal so cactus clusters become one block."""
    if gap_px <= 1:
        return col_hit.astype(bool)
    a = col_hit.astype(np.uint8)
    k = np.ones((gap_px,), np.uint8)
    a = cv2.dilate(a, k, iterations=1)
    a = cv2.erode(a, k, iterations=1)
    return a.astype(bool)


def detect_next_obstacle_block(game_frame):
    """
    Returns None or dict:
      {"lead_x": int, "trail_x": int, "width_px": int, "rect": (x1,y1,x2,y2)}
    All x are in GAME-FRAME coordinates.

    Uses:
      - occupancy fraction (less biased by bold single cactus)
      - 1D closing to bridge forest gaps
    """
    roi, rect = get_roi(game_frame, LOOK_X_REL, LOOK_W, LOOK_Y_OFF, LOOK_H)
    if roi.size == 0:
        return None

    occ = _column_occupancy_frac(roi, thr=DARK_THR, invert=args.invert)
    col_hit = occ >= OCC_THRESH
    col_hit = _close_1d(col_hit, gap_px=GAP_PX)

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
    """Wider obstacle => jump earlier (smaller x)."""
    return LARGE_JUMP_X if width_px > LARGE_PX else SMALL_JUMP_X


# =============================================================================
# NEW: "LAND JUST BEHIND OBSTACLE" TRACKING HELPERS
# =============================================================================
def start_obstacle_tracking(obs, now):
    """
    Call this RIGHT WHEN YOU JUMP.

    We "lock" onto the obstacle we jumped for by remembering its trailing edge.
    As frames progress, we keep updating trail_x while it is visible.
    Once trail_x is behind SAFE_CLEAR_X, we fast-drop to land ASAP behind it.
    """
    return {
        "active": True,
        "trail_x_last": obs["trail_x"],
        "jump_t": now,
        "drop_done": False,
    }


def update_obstacle_tracking(track, obs):
    """
    Each frame, update trailing edge while tracking is active.
    If detection flickers, we keep the last known trail_x.
    """
    if track is None or not track.get("active", False):
        return track

    if obs is not None:
        track["trail_x_last"] = obs["trail_x"]

    return track


def should_fast_drop_to_land_behind(track, now):
    """
    Safe/useful to drop only when:
      1) We've been airborne long enough (MIN_AIR_TIME)
      2) The obstacle's trailing edge is behind SAFE_CLEAR_X
      3) We haven't already done the drop for this obstacle
    """
    if track is None or not track.get("active", False):
        return False

    if track.get("drop_done", False):
        return False

    if (now - track["jump_t"]) < MIN_AIR_TIME:
        return False

    return track["trail_x_last"] < SAFE_CLEAR_X


def finish_obstacle_tracking(track):
    """Stop tracking after we fast-drop once."""
    if track is not None:
        track["active"] = False
    return track


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

    # State to avoid repeated jumps on same obstacle
    armed = True
    in_air = False

    # NEW: Track the obstacle we jumped for so we can fast-drop right after its trailing edge clears.
    track = None

    with mss.mss() as sct:
        while True:
            now = time.time()

            game_img = np.array(sct.grab(monitor))
            game_frame = game_img[:, :, :3].copy()
            h, w = game_frame.shape[:2]

            obs = detect_next_obstacle_block(game_frame)

            # -----------------------------------------------------------------
            # (existing) rough airborne timer
            # NOTE: you might want this longer (0.25–0.40) depending on jump arc.
            # -----------------------------------------------------------------
            if in_air and (now - last_jump_t) > 0.1:
                in_air = False

            # -----------------------------------------------------------------
            # NEW: update tracking every frame (keeps trail_x_last fresh)
            # -----------------------------------------------------------------
            track = update_obstacle_tracking(track, obs)

            # Re-arm once the current obstacle is clearly behind us (your existing logic)
            if obs is not None:
                if obs["trail_x"] < SAFE_CLEAR_X:
                    armed = True
            else:
                armed = True

            # -----------------------------------------------------------------
            # JUMP DECISION (unchanged except cooldown + starting tracking)
            # -----------------------------------------------------------------
            if obs is not None:
                trig_x = trigger_x_for_width(obs["width_px"])
                can_jump = armed

                if can_jump and obs["lead_x"] <= trig_x:
                    jump()
                    last_jump_t = now
                    in_air = True
                    armed = False

                    # =========================
                    # NEW: start tracking THIS obstacle so we can land right behind it
                    # =========================
                    track = start_obstacle_tracking(obs, now)

            # -----------------------------------------------------------------
            # NEW: LAND-JUST-BEHIND logic
            # Instead of dropping whenever, we drop ONCE the trailing edge clears SAFE_CLEAR_X.
            # This makes you land as early as possible behind the last cactus.
            # -----------------------------------------------------------------
            if in_air and should_fast_drop_to_land_behind(track, now):
                fast_drop(DROP_HOLD)
                track["drop_done"] = True
                track = finish_obstacle_tracking(track)

            # -----------------------------------------------------------------
            # DEBUG overlays
            # -----------------------------------------------------------------
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

            # NEW: show tracking info
            if track is not None and track.get("active", False):
                cv2.putText(
                    game_frame,
                    f"track trail_x_last={track['trail_x_last']}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
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
