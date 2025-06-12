import os
import argparse
import cv2
import moviepy.editor
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)

'''
This script extracts frames from SoccerNetv2 Ball Action Spotting dataset by introducing the path where the downloaded videos are (at 720p resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution.

python extract_frames_snb.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 25 --num_workers 5
'''

FRAME_CORRECT_THRESHOLD = 1000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=int, default=25)
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)
    parser.add_argument('--target_height', type=int, default=448)
    parser.add_argument('--target_width', type=int, default=796)
    parser.add_argument('--original_resolution', type=str, default='720p')
    return parser.parse_args()


def get_duration(video_path):
    return moviepy.editor.VideoFileClip(video_path).duration

def extract_frames_opencv(mp4_path, output_dir, target_fps=25, width=796, height=448):
    """
    Extract frames from an MP4 at a target FPS.
    Returns the total number of frames written.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / source_fps

    # Calculate time interval between frames to save
    target_interval_sec = 1.0 / target_fps
    next_capture_time = 0.0

    frames_written = 0

    pbar = tqdm(total=int(duration_sec * target_fps), desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_time_sec >= next_capture_time:
            frame_resized = cv2.resize(frame, (width, height))
            out_name = os.path.join(output_dir, f"frame{frames_written+1}.jpg")
            cv2.imwrite(out_name, frame_resized)
            frames_written += 1
            next_capture_time += target_interval_sec
            pbar.update(1)

    pbar.close()
    cap.release()
    return frames_written

def worker(args):
    video_name, video_path, out_dir, sample_fps = args

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Extracting frames from {video_name} with target {sample_fps} fps...")

    frames_written = extract_frames_opencv(
        mp4_path=video_path,
        output_dir=out_dir,
        target_fps=sample_fps,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )

    print(f"{video_name} - done, {frames_written} frames written.")



def main(args):
    video_dir = args.video_dir
    out_dir = args.out_dir
    sample_fps = args.sample_fps
    num_workers = args.num_workers
    target_height = args.target_height
    target_width = args.target_width
    original_resolution = args.original_resolution

    global TARGET_HEIGHT
    TARGET_HEIGHT = target_height
    global TARGET_WIDTH
    TARGET_WIDTH = target_width

    out_dir = out_dir + str(TARGET_HEIGHT)

    worker_args = []
    for game in os.listdir(video_dir):
        game_dir = os.path.join(video_dir, game)
        for video_file in os.listdir(game_dir):
            if (video_file.endswith(original_resolution + '.mp4') | video_file.endswith(original_resolution + '.mkv')):
                worker_args.append((
                    os.path.join(game, video_file),
                    os.path.join(game_dir, video_file),
                    os.path.join(
                        out_dir, game
                    ) if out_dir else None,
                    sample_fps
                ))

    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args),
                    total=len(worker_args)):
            pass
    print('Done!')


if __name__ == '__main__':
    args = get_args()
    main(args)