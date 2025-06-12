import cv2
import json
import math

###############################################################################
#                         UTILITY FUNCTIONS
###############################################################################
def format_game_time(milliseconds):
    """
    Convert milliseconds to '1 - mm:ss' format with two-digit minutes and seconds.
    """
    total_seconds = int(milliseconds // 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"1 - {minutes:02}:{seconds:02}"

###############################################################################
#                         CONFIG
###############################################################################
FOLDER_PATH = "/mnt/d/soccernetBAS"
GAME_PATH = "/RBK-kamper/Viking-RBK/game.mp4"
VIDEO_PATH = f"{FOLDER_PATH}{GAME_PATH}"

INTERVAL = 6
INTERVAL_LENGTH_MINUTES = 15

URL_LOCAL   = GAME_PATH
URL_YOUTUBE = ""
HALFTIME    = "ikke satt"

# Jump amounts
NORMAL_JUMP_SECONDS = 3     
ANNOTATE_JUMP_MS    = 40     

# Playback speed bounds
PLAYBACK_SPEED_MIN = 0.25
PLAYBACK_SPEED_MAX = 4.0

EVENT_KEY_MAP = {
    ord('1'): "PASS",
    ord('2'): "CROSS",
    ord('3'): "THROW IN",
    ord('4'): "SHOT",
    ord('5'): "FREE KICK",
    ord('6'): "Corner",
    ord('7'): "Kickoff",
    ord('8'): "SUCCESSFUL THROUGH BALL",
}

TEAM_KEY_MAP = {
    ord('w'): "left",
    ord('e'): "right",
    ord('r'): "none",
}
# Additional key codes
QUIT_KEY       = ord('q')
SPACE_KEY      = ord(' ')
RIGHT_ARROW    = 83
LEFT_ARROW     = 81
SPEED_UP_KEY   = 82  # Up arrow
SLOW_DOWN_KEY  = 84  # Down arrow
BACKSPACE_KEY  = 8   # Backspace

###############################################################################
#                          HELPER: KEYMAP DISPLAY
###############################################################################
# We build a small instructions list to display on screen
HELP_LINES = [
    "EVENT KEYS:",
    " 1: PASS",
    " 2: CROSS",
    " 3: THROW IN",
    " 4: SHOT",
    " 5: FREE KICK",
    " 6: Corner",
    " 7: Kickoff",
    " 8: SUCCESSFUL THROUGH BALL",
    "",
    "TEAM KEYS:",
    "  w: left",
    "  e: right",
    "  r: none",
]

###############################################################################
#                       MAIN ANNOTATION SCRIPT
###############################################################################
def main():
    # Create named window in normal mode, then set full screen
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Video Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Inside your loop, before cv2.imshow(...), do:
    screen_width = 1536
    screen_height = 864


    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open: {VIDEO_PATH}")
        return

    # 1) Detect the FPS from file
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
        print("Warning: No valid FPS found, defaulting to 30 fps.")

    # 2) Calculate start/end in ms for the chosen interval
    start_ms = (INTERVAL - 1) * INTERVAL_LENGTH_MINUTES * 60_000
    end_ms   = INTERVAL * INTERVAL_LENGTH_MINUTES * 60_000

    # Move to start of interval
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    # Read one frame to begin
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video or failed to read the start.")
        return

    # Or get your actual screen size programmatically (e.g., using PyQt, tk, or Win32 APIs).
    # For demonstration, let's assume 1920x1080:



    paused = False
    annotating_mode = False
    selected_event = None
    playback_speed = 1.0

    annotations = []

    print(f"Detected FPS: {fps:.2f}")
    print(f"Annotating interval #{INTERVAL}: from {start_ms}ms to {end_ms}ms.")
    print("Controls:")
    print("  SPACE: pause/resume (unless annotating)")
    print("  f / Right Arrow: jump forward 5s (normal) or small ms (annotating)")
    print("  b / Left Arrow : jump backward 5s (normal) or small ms (annotating)")
    print("  Up/Down Arrows : speed up / slow down (while playing)")
    print("  1..6           : pick an event => enters 'annotating mode' & auto-paused")
    print("  z/x/m          : pick a team => logs annotation & resumes normal mode")
    print("  BACKSPACE      : cancel annotation & resume normal mode")
    print("  q              : quit (save)")
    print("-----------------------------------------------------------")

    while True:
        # If not paused, read next frame
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or can't read further.")
                break

        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_ms < 0:
            print("Cannot read current ms from capture. Exiting.")
            break

        # Check if we've hit end of interval
        if current_ms >= end_ms:
            print("Reached end of this interval.")
            break

        # Build overlay (Time & Speed)
        time_str = format_game_time(current_ms)
        overlay_text = f"Time: {time_str}  Speed: {playback_speed:.2f}x"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # If annotating mode, show event bottom-right
        if annotating_mode and selected_event:
            text_event = f"Annotating: {selected_event}"
            (text_w, text_h), _ = cv2.getTextSize(text_event, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x = frame.shape[1] - text_w - 10
            y = frame.shape[0] - 10
            cv2.putText(frame, text_event, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Show help lines in top-left corner, a bit below the main overlay
        line_x, line_y = 10, 60
        font_scale = 0.6
        for i, line in enumerate(HELP_LINES):
            y_pos = line_y + i * 20
            cv2.putText(frame, line, (line_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
            
        #resized_frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

        # Show frame
        cv2.imshow("Video Player", frame)

        # Decide waitKey
        if paused or annotating_mode:
            delay = 30
        else:
            base_delay = 1000 / fps
            delay = max(1, int(base_delay / playback_speed))

        key = cv2.waitKey(delay) & 0xFF

        if key == QUIT_KEY:  # 'q'
            print("Quitting.")
            break

        elif key == SPACE_KEY:
            if annotating_mode:
                print("Ignoring SPACE in annotating mode (still paused).")
            else:
                paused = not paused
                print("Paused" if paused else "Resumed")

        elif key == SPEED_UP_KEY:  # Up arrow => speed up
            if not paused and not annotating_mode:
                playback_speed = min(playback_speed * 2.0, PLAYBACK_SPEED_MAX)
                print(f"Speed => {playback_speed}x")

        elif key == SLOW_DOWN_KEY: # Down arrow => slow down
            if not paused and not annotating_mode:
                playback_speed = max(playback_speed / 2.0, PLAYBACK_SPEED_MIN)
                print(f"Speed => {playback_speed}x")

        elif key == ord('f') or key == RIGHT_ARROW:
            # forward jump
            if annotating_mode:
                jump_ms = ANNOTATE_JUMP_MS
            else:
                jump_ms = NORMAL_JUMP_SECONDS * 1000
            new_ms = current_ms + jump_ms
            if new_ms > end_ms:
                new_ms = end_ms - 1
            if new_ms < 0:
                new_ms = 0

            cap.set(cv2.CAP_PROP_POS_MSEC, new_ms)
            ret, frame = cap.read()
            if ret:
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"Jump forward {jump_ms} ms -> now at {int(current_ms)} ms.")
            else:
                print("Error reading after jump.")
                break

        elif key == ord('b') or key == LEFT_ARROW:
            # backward jump
            if annotating_mode:
                jump_ms = ANNOTATE_JUMP_MS
            else:
                jump_ms = NORMAL_JUMP_SECONDS * 1000
            new_ms = current_ms - jump_ms
            if new_ms < start_ms:
                new_ms = start_ms
            if new_ms < 0:
                new_ms = 0

            cap.set(cv2.CAP_PROP_POS_MSEC, new_ms)
            ret, frame = cap.read()
            if ret:
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"Jump backward {jump_ms} ms -> now at {int(current_ms)} ms.")
            else:
                print("Error reading after jump.")
                break

        elif key in EVENT_KEY_MAP:
            # Enter annotating mode
            selected_event = EVENT_KEY_MAP[key]
            paused = True
            annotating_mode = True
            print(f"Selected event '{selected_event}'. Annotating mode ON (paused).")

        elif annotating_mode and selected_event and (key in TEAM_KEY_MAP):
            # Finalize annotation
            selected_team = TEAM_KEY_MAP[key]
            annotation = {
                "gameTime": format_game_time(current_ms),
                "label": selected_event,
                "position": str(int(current_ms)),
                "team": selected_team,
                "visibility": "visible"
            }
            annotations.append(annotation)
            print(f"Logged annotation: {annotation}")

            # Exit mode
            selected_event = None
            annotating_mode = False
            paused = False
            print("Resuming playback...")

        elif annotating_mode and key == BACKSPACE_KEY:
            print("Annotation canceled. Exiting annotating mode.")
            selected_event = None
            annotating_mode = False
            paused = False

        # else: unrecognized key, ignore

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Build JSON
    output_dict = {
        "UrlLocal": URL_LOCAL,
        "UrlYoutube": URL_YOUTUBE,
        "halftime": HALFTIME,
        "annotations": annotations
    }

    out_filename = f"annotations_interval{INTERVAL}.json"
    with open(out_filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    print(f"Saved {len(annotations)} annotations to '{out_filename}'.")


if __name__ == "__main__":
    main()
