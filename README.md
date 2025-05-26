# Watermark Remover for Bushnell Trail Camera Videos

Bushnell trail cameras add a bright orange 110x110 pixel Bushnell logo to the lower left corner of videos. This logo is obtrusive and there is no way to politely ask the camera to not do this.  So here we have a cross-platform Python tool which will automatically obscure this watermark. It does this by whiting-out the lower part of the logo (to match the info bar across the bottom of the frame) and pasting over the top half of the logo with a piece of nearby image. The script extracts frames, patches the watermark region, and reassembles the cleaned video.

## Features
- Removes Bushnell watermark from video frames
- Fast, parallelized frame processing
- Customizable patch size and position
- Progress bar with ETA
- Automatic cleanup of temporary files
- Cross-platform (Linux, macOS, Windows)

## Requirements
- Python 3.7+
- ffmpeg and ffprobe (must be installed and in your PATH)
- Python packages: opencv-python, numpy, tqdm

## Installation
1. Install ffmpeg (see https://ffmpeg.org/download.html)
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
```sh
python watermark_remover.py input_video.mp4
```

### Options
- `-o`, `--output`         Output video file (default: <input>_cleaned.mp4)
- `--patch-width`          Patch width (default: 110)
- `--patch-height`         Patch height (default: 110)
- `--patch-x`              Patch X offset (default: 0)
- `--patch-y`              Patch Y offset (default: 0)
- `--mirror-height`        Height of mirrored patch (default: 54)
- `--mirror-offset`        Offset above patch for mirrored region (default: 56)
- `--tmpdir`               Temporary directory (default: frames_<input name>)
- `--keep-temp`            Keep temporary frames directory
- `-j`, `--jobs`           Number of parallel jobs (default: all cores)

### Example
```sh
python watermark_remover.py myvideo.mp4 -o output.mp4 --patch-width 120 --patch-height 100
```

## Troubleshooting
- **ffmpeg not found:** Make sure ffmpeg and ffprobe are installed and in your PATH.
- **Missing Python packages:** Install with `pip install -r requirements.txt`.
- **Output video is empty or corrupted:** Check that the input video is valid and supported by ffmpeg.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
