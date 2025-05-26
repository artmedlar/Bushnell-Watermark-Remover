#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
import subprocess
import time
import glob
from pathlib import Path
import functools

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        print("Error: ffmpeg and/or ffprobe not found in PATH.")
        sys.exit(1)

def get_fps(input_path):
    """Get the frame rate of the input video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, denom = map(float, fps_str.split("/"))
        return num / denom if denom else num
    return float(fps_str)

def extract_frames(input_path, tmpdir):
    """Extract frames from video using ffmpeg."""
    os.makedirs(tmpdir, exist_ok=True)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", input_path,
        f"{tmpdir}/frame_%05d.png"
    ]
    subprocess.run(cmd, check=True)

def patch_frame(fname, input_dir, patch_width, patch_height, patch_x, patch_y, mirror_height, mirror_offset):
    """
    Patches a single frame to remove a Bushnell trail camera watermark.

    The watermark is a 110x110 pixel bright orange square located in the
    lower-left corner of the video, extending to the bottom of the frame.
    A 56-pixel high white bar containing time/date information is typically present
    across the bottom of the frame elsewhere, but *not* under the orange square itself.

    The patching strategy is two-fold:
    1. Lower Part (typically 56 pixels high, matching the white info bar):
       A 110x56 region is copied from the area immediately to the right
       of the watermark's lower section. This helps match any video
       post-processing like fade-ins that might affect the white bar's brightness.
    2. Upper Part (typically 54 pixels high, the remainder of the watermark):
       A 110x54 region directly above the watermark is mirrored vertically
       and used to cover this portion.

    This per-frame processing accounts for camera movement and lighting changes,
    resulting in a non-noticeable patch in typical greyscale outdoor scenes.

    Args:
        fname (str): Filename of the frame to patch.
        input_dir (str): Directory containing the frame.
        patch_width (int): Width of the watermark area (default 110).
        patch_height (int): Total height of the watermark area (default 110).
        patch_x (int): X-coordinate of the bottom-left corner of the watermark (default 0).
        patch_y (int): Y-coordinate of the bottom-left corner of the watermark,
                       measured from the bottom of the frame (default 0).
        mirror_height (int): Height of the content to be mirrored for the upper part
                             of the patch (default 54).
        mirror_offset (int): Vertical offset *above* the watermark area from where
                             the mirrored content is sourced (default 56).
                             This means the source for mirroring starts `mirror_offset` pixels
                             above the top of the `patch_height` area defined by `patch_y`.

    Returns:
        bool: True if patching was successful, False otherwise.
    """
    path = os.path.join(input_dir, fname)
    frame = cv2.imread(path)
    if frame is None:
        return False
    height, width, _ = frame.shape
    if height < patch_height or width < patch_width + patch_x:
        return False
    
    # Watermark area to be replaced (bottom-left origin, patch_y from bottom of frame)
    # wm_y_start is the top Y of the patch area, wm_y_end is the bottom Y.
    wm_y_start = height - (patch_y + patch_height) # Top of the watermark area
    wm_y_end = height - patch_y                  # Bottom of the watermark area (e.g. bottom of frame if patch_y=0)
    wm_x_start = patch_x                         # Left of the watermark area
    wm_x_end = patch_x + patch_width             # Right of the watermark area

    # Source for the upper, mirrored part of the patch
    # This content is taken from directly above the watermark area.
    # mirror_offset defines how many pixels *above* the watermark's top edge
    # the source for the mirrored content begins.
    mirror_src_y_start = wm_y_start - mirror_offset # Top of the source region for mirroring
    mirror_src_y_end = wm_y_start - mirror_offset + mirror_height # Bottom of the source region for mirroring
    mirror_src_x_start = patch_x # Source X is aligned with watermark X
    mirror_src_x_end = patch_x + patch_width
    
    # Source for the lower, adjacent part of the patch
    # This content is typically taken from the area immediately to the right
    # of the lower part of the watermark (i.e., within the white info bar).
    # Its height is (patch_height - mirror_height), which is 110 - 54 = 56 pixels by default.
    adjacent_patch_height = patch_height - mirror_height
    # Y-coordinates for the source of this adjacent patch should align with the lower part of the watermark area.
    adj_src_y_start = wm_y_end - adjacent_patch_height # Top of adjacent source (aligns with top of where it will go)
    adj_src_y_end = wm_y_end                         # Bottom of adjacent source (aligns with bottom of patch area)
    adj_src_x_start = patch_x + patch_width          # Start immediately to the right of the watermark area
    adj_src_x_end = patch_x + 2 * patch_width      # End patch_width pixels further to the right

    # Defensive checks for coordinates (simple version)
    if not (0 <= mirror_src_y_start < mirror_src_y_end <= height and \
            0 <= mirror_src_x_start < mirror_src_x_end <= width and \
            0 <= adj_src_y_start < adj_src_y_end <= height and \
            0 <= adj_src_x_start < adj_src_x_end <= width and \
            0 <= wm_y_start < wm_y_end <= height and \
            0 <= wm_x_start < wm_x_end <= width):
        return False

    mirrored_content = frame[mirror_src_y_start:mirror_src_y_end, mirror_src_x_start:mirror_src_x_end]
    mirrored_content_flipped = cv2.flip(mirrored_content, 0)

    # Ensure flipped mirrored content has the correct height for vconcat
    if mirrored_content_flipped.shape[0] != mirror_height:
        return False

    adjacent_content = frame[adj_src_y_start:adj_src_y_end, adj_src_x_start:adj_src_x_end]
    
    # Ensure adjacent content has correct height
    if adjacent_content.shape[0] != adjacent_patch_height:
        return False

    # Combine them
    if mirrored_content_flipped.shape[1] != patch_width or adjacent_content.shape[1] != patch_width:
        return False

    try:
        patch = cv2.vconcat([mirrored_content_flipped, adjacent_content])
    except cv2.error as e:
        return False

    if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
        return False

    frame[wm_y_start:wm_y_end, wm_x_start:wm_x_end] = patch
    cv2.imwrite(path, frame)
    return True

def _global_patch_frame_wrapper(fname, input_dir, patch_width, patch_height, patch_x, patch_y, mirror_height, mirror_offset):
    """
    Top-level wrapper for patch_frame to be used with multiprocessing.
    All arguments must be picklable.
    """
    return patch_frame(fname, input_dir, patch_width, patch_height, patch_x, patch_y, mirror_height, mirror_offset)

def patch_frames(tmpdir, patch_width, patch_height, patch_x, patch_y, mirror_height, mirror_offset, n_jobs):
    frame_files = sorted([f for f in os.listdir(tmpdir) if f.endswith(".png")])
    total = len(frame_files)
    if total == 0:
        print("⚠️ No frames found to patch.")
        return

    print(f"🛠️  Patching {total} frames with {n_jobs} workers...")

    # Use functools.partial to create a picklable worker function
    # with fixed arguments (all except the one that varies per call, which is fname)
    worker_fn = functools.partial(
        _global_patch_frame_wrapper,
        input_dir=tmpdir,
        patch_width=patch_width,
        patch_height=patch_height,
        patch_x=patch_x,
        patch_y=patch_y,
        mirror_height=mirror_height,
        mirror_offset=mirror_offset
    )

    # process_map will pass each item from frame_files as the first argument to worker_fn
    results = process_map(
        worker_fn,
        frame_files,
        max_workers=n_jobs,
        desc="Patching frames",
        unit="frame",
        chunksize=1
    )
    
    successful_patches = sum(1 for r in results if r is True)
    print(f"✅ {successful_patches}/{total} frames patched successfully.")

def assemble_video(tmpdir, output_path, fps):
    if not glob.glob(f"{tmpdir}/frame_*.png"):
        print(f"⚠️ No frames found in {tmpdir}. Skipping video assembly.")
        return

    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", f"{tmpdir}/frame_%05d.png",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryslow", "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during video assembly: {e}")
        print("Make sure ffmpeg is installed and the frames in the temporary directory are valid.")

def main():
    parser = argparse.ArgumentParser(description="Remove Bushnell trail camera watermark from video.")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (default: <input>_cleaned.mp4)")
    parser.add_argument("--patch-width", type=int, default=110, help="Patch width (default: 110)")
    parser.add_argument("--patch-height", type=int, default=110, help="Patch height (default: 110)")
    parser.add_argument("--patch-x", type=int, default=0, help="Patch X offset (bottom-left corner of patch area) (default: 0)")
    parser.add_argument("--patch-y", type=int, default=0, help="Patch Y offset (bottom-left corner of the watermark area, measured from the bottom of the frame) (default: 0)")
    parser.add_argument("--mirror-height", type=int, default=54, help="Height of the content that will be mirrored to form the upper part of the patch (default: 54 for Bushnell)")
    parser.add_argument("--mirror-offset", type=int, default=56, help="Vertical offset (pixels) *above* the main watermark area from where the source content for mirroring is taken (default: 56 for Bushnell, typically above the orange square, within the actual video content)")
    parser.add_argument("--tmpdir", help="Temporary directory (default: frames_<input name>)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary frames directory")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Number of parallel jobs (default: all cores)")
    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    name = input_path.stem
    output_path = Path(args.output or f"{name}_cleaned{input_path.suffix}")
    tmpdir = Path(args.tmpdir or f"frames_{name}")

    start_time = time.time()
    
    print(f"⏳ Starting watermark removal for {input_path}...")
    print(f"⚙️  Parameters: Patch(W:{args.patch_width}, H:{args.patch_height}, X:{args.patch_x}, Y:{args.patch_y}), Mirror(H:{args.mirror_height}, Offset:{args.mirror_offset})")
    print(f"🕒 Using {args.jobs} worker(s). Output to: {output_path}, Temp dir: {tmpdir}")

    print(f"📸 Extracting frames from {input_path} to {tmpdir}...")
    extract_frames(str(input_path), str(tmpdir))

    fps = get_fps(str(input_path))
    print(f"🎞️  Detected video frame rate: {fps:.2f} fps")

    patch_frames(
        str(tmpdir),
        args.patch_width,
        args.patch_height,
        args.patch_x,
        args.patch_y,
        args.mirror_height,
        args.mirror_offset,
        args.jobs
    )

    print(f"🎞️  Encoding final video to {output_path} at {fps:.2f} fps...")
    assemble_video(str(tmpdir), str(output_path), fps)

    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"🕒 Total processing time: {int(elapsed // 60)}m {int(elapsed % 60):02d}s")
    
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"✅ Done! Output written to: {output_path}")
    else:
        print(f"❌ Output file {output_path} was not created or is empty.")

    if not args.keep_temp:
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
            print(f"🧹 Temporary directory '{tmpdir}' removed.")
    else:
        print(f"🗂️  Temporary frames kept in '{tmpdir}'.")

if __name__ == "__main__":
    main() 