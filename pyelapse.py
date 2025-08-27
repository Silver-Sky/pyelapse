import os
import calendar
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import click
import subprocess
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import exifread
# Enable HEIC support via Pillow if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF_ENABLED = True
except Exception:
    _HEIF_ENABLED = False

from datetime import datetime, timedelta
import concurrent.futures
import numpy as np
 


def _check_ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None

@click.group()
def cli():
    """PyElapse CLI - Create time-lapse videos from images."""
    pass


# -----------------------------
# Utility helpers (fast paths)
# -----------------------------

ALLOWED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".heic", ".raw", ".cr2", ".nef", ".arw"
)


def list_image_paths(folder: Path, allowed_exts: Sequence[str]) -> List[str]:
    return sorted(
        [str(folder / f) for f in os.listdir(folder) if f.lower().endswith(tuple(allowed_exts))]
    )


def list_images_recursive(folder: Path, allowed_exts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(tuple(allowed_exts)):
                out.append(str(Path(root) / f))
    return out


def read_exif_datetime(path: str) -> Optional[datetime]:
    try:
        suffix = os.path.splitext(path)[1].lower()
        # HEIC: use Pillow EXIF because exifread doesn't support HEIC containers
        if suffix == ".heic":
            try:
                with Image.open(path) as im:
                    exif = im.getexif()
                    if exif:
                        # 36867: DateTimeOriginal, 306: Image DateTime
                        for tag_id in (36867, 306):
                            val = exif.get(tag_id)
                            if val:
                                dt_str = str(val)
                                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            except Exception:
                return None
        # Default path for JPEG/TIFF and others supported by exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(
                f, stop_tag="EXIF DateTimeOriginal", details=False
            )
            dt_str = str(tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime"))
        if dt_str and dt_str != "None":
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def precompute_timestamps(images: Sequence[str]) -> List[str]:
    """Read EXIF datetimes for images in parallel and return printable strings."""
    results: List[Optional[datetime]] = [None] * len(images)

    def worker(idx_img: Tuple[int, str]) -> Tuple[int, Optional[datetime]]:
        idx, img_path = idx_img
        return idx, read_exif_datetime(img_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8) * 2)) as executor:
        for idx, dt in executor.map(worker, list(enumerate(images))):
            results[idx] = dt

    out: List[str] = []
    for img_path, dt in zip(images, results):
        if dt is not None:
            out.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            filename = os.path.basename(img_path)
            if any(ch.isdigit() for ch in filename):
                out.append(filename)
            else:
                out.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return out


def parse_normalized_filename_dt(path: Path) -> Optional[datetime]:
    """Parse timestamps from normalized filenames like YYYY-MM-DD-HH-MM-SS[-counter].ext"""
    try:
        stem = path.stem
        parts = stem.split("-")
        # Expect at least 6 parts for YYYY MM DD HH MM SS
        if len(parts) < 6:
            return None
        base_parts = parts[:6]
        base_str = "-".join(base_parts)
        return datetime.strptime(base_str, "%Y-%m-%d-%H-%M-%S")
    except Exception:
        return None


def find_last_normalized_dt(output_folder: Path) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for p in output_folder.iterdir():
        if p.is_file():
            dt = parse_normalized_filename_dt(p)
            if dt is not None and (latest is None or dt > latest):
                latest = dt
    return latest


def get_codec_for_output(output: str) -> Tuple[int, str]:
    """Choose fourcc and human-readable codec name from output extension."""
    ext = os.path.splitext(output)[1].lower()
    if ext == ".mov":
        return cv2.VideoWriter_fourcc(*"avc1"), "H.264 (avc1) in .mov"
    if ext in [".mp4", ".m4v"]:
        return cv2.VideoWriter_fourcc(*"mp4v"), "H.264 (mp4v) in .mp4/.m4v"
    if ext == ".mkv":
        return cv2.VideoWriter_fourcc(*"X264"), "H.264 (X264) in .mkv"
    return cv2.VideoWriter_fourcc(*"mp4v"), "H.264 (mp4v) default"


def draw_timestamp_on_frame(
    frame: np.ndarray,
    text: str,
    margin: int = 10,
    box_alpha: float = 0.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Fast OpenCV overlay of semi-transparent box and timestamp text in lower-right."""
    height, width = frame.shape[:2]

    # Auto scale font relative to width
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.5, width / 1600.0))
    thickness = max(1, int(round(2 * scale)))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = width - text_w - margin
    y = height - margin

    # Background rectangle coordinates
    top_left = (x - 6, y - text_h - 6)
    bottom_right = (x + text_w + 6, y + baseline + 4)

    # Blend semi-transparent rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, box_color, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, dst=frame)

    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    return frame


def imread_any(img_path: str) -> Optional[np.ndarray]:
    """Read image from path using Pillow to honor EXIF Orientation, then convert to BGR ndarray.
    Falls back to OpenCV if Pillow fails.
    """
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
            arr = np.array(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback for formats Pillow can't read
        return cv2.imread(img_path)

@cli.command('create-timelapse')
@click.argument('folder', type=click.Path(exists=True, file_okay=False))
@click.option('--fps', default=24, show_default=True, help='Frames per second for the output video.')
@click.option('--output', default='output.mov', show_default=True, help='Output video file name (H.264 in .mov container).')
@click.option('--crf', type=int, default=None, help='Compress output using ffmpeg with given CRF (lower is better, 18-28).')
@click.option('--timestamp', is_flag=True, default=False, help='Add timestamp overlay to the lower right corner of each frame.')
def create_timelapse(folder, fps, output, crf, timestamp):
    """
    Create a time-lapse video from images in a folder.
    :param folder: Path to the folder containing images.
    :param fps: frames per second for the output video.
    :param output: filename for the output video.
    :param crf: compression quality for ffmpeg (optional).
    :param timestamp: whether to add timestamp overlay to frames.
    :return: None
    """
    # --- Check for ffmpeg availability early ---
    if not _check_ffmpeg_installed():
        raise click.ClickException("ffmpeg is not installed or not found in PATH. Please install it to use video features.")

    folder_path = Path(folder)
    images = list_image_paths(folder_path, (".png", ".jpg", ".jpeg", ".heic"))
    if not images:
        click.secho('No images found in the folder.', fg='red')
        return

    # Read first readable frame to determine dimensions
    frame = None
    first_readable_path: Optional[str] = None
    for p in images:
        frame = imread_any(p)
        if frame is not None:
            first_readable_path = p
            break
    if frame is None:
        click.secho('No readable images found (unsupported formats?).', fg='red')
        return
    height, width = frame.shape[:2]

    # Select codec based on file extension
    fourcc, codec_name = get_codec_for_output(output)

    click.secho("=== Timelapse Creation Details ===", fg='cyan', bold=True)
    click.secho(f"Images used: {len(images)}", fg='blue')
    click.secho(f"Resolution: {width}x{height}", fg='blue')
    click.secho(f"FPS: {fps}", fg='blue')
    click.secho(f"Codec: {codec_name}", fg='blue')
    click.secho(f"Output file: {output}", fg='blue')
    click.secho(f"First image: {os.path.basename(images[0])}", fg='blue')
    click.secho(f"Last image: {os.path.basename(images[-1])}", fg='blue')
    if timestamp:
        click.secho("Timestamps: Enabled", fg='blue')
    click.secho("=" * 32, fg='cyan')

    output_path = Path(output)
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")

    # Determine incremental start index using state
    start_idx = 0
    last_image_path: Optional[str] = None
    if state_path.exists():
        try:
            import json as _json
            with open(state_path, "r", encoding="utf-8") as f:
                st = _json.load(f)
                last_image_path = st.get("last_image")
        except Exception:
            last_image_path = None
    if last_image_path and last_image_path in images:
        try:
            start_idx = images.index(last_image_path) + 1
        except ValueError:
            start_idx = 0

    images_to_process = images[start_idx:] if start_idx < len(images) else []

    if not output_path.exists():
        # Fresh build: write all frames
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Precompute timestamps for all images if needed
        timestamps: Optional[List[str]] = None
        if timestamp:
            timestamps = precompute_timestamps(images)

        with click.progressbar(images, label="Creating timelapse") as bar:
            for idx, img_path in enumerate(bar):
                img = imread_any(img_path)
                if img is None:
                    continue
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                if timestamp and timestamps is not None:
                    img = draw_timestamp_on_frame(img, timestamps[idx])
                out.write(img)
        out.release()
        click.secho(f"Time-lapse video saved as {output}", fg='green', bold=True)
    else:
        # Append only new frames, then concat with existing video
        if not images_to_process:
            click.secho("No new images to append.", fg='yellow')
        else:
            part_output = output_path.with_stem(output_path.stem + "_part")
            out_part = cv2.VideoWriter(str(part_output), fourcc, fps, (width, height))

            timestamps_part: Optional[List[str]] = None
            if timestamp:
                timestamps_part = precompute_timestamps(images_to_process)

            with click.progressbar(images_to_process, label="Appending frames") as bar:
                for idx_rel, img_path in enumerate(bar):
                    img = imread_any(img_path)
                    if img is None:
                        continue
                    if img.shape[:2] != (height, width):
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                    if timestamp and timestamps_part is not None:
                        img = draw_timestamp_on_frame(img, timestamps_part[idx_rel])

                    out_part.write(img)
            out_part.release()

            # Concat existing output with part using ffmpeg stream copy
            concat_list = output_path.with_suffix(output_path.suffix + ".concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                f.write(f"file '{str(output_path).replace("'", "'\\''")}'\n")
                f.write(f"file '{str(part_output).replace("'", "'\\''")}'\n")

            # Keep the original container extension so ffmpeg can infer the muxer (e.g., .tmp.mov instead of .mov.tmp)
            merged_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-c", "copy", str(merged_output)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and merged_output.exists():
                # Replace original
                try:
                    os.replace(str(merged_output), str(output_path))
                    click.secho(f"Appended {len(images_to_process)} frames to {output}", fg='green', bold=True)
                finally:
                    try:
                        os.remove(concat_list)
                    except Exception:
                        pass
                    try:
                        os.remove(part_output)
                    except Exception:
                        pass
            else:
                click.secho(f"Concat failed: {result.stderr.decode()}", fg='red')
                # Cleanup partials
                try:
                    os.remove(concat_list)
                except Exception:
                    pass
                try:
                    os.remove(part_output)
                except Exception:
                    pass

    # Update state
    try:
        import json as _json
        last_img = images[-1] if images else None
        if last_img:
            with open(state_path, "w", encoding="utf-8") as f:
                _json.dump({"last_image": last_img}, f)
    except Exception:
        pass

    # Compress with ffmpeg if requested
    # Compress with ffmpeg if requested
    if crf is not None:
        if not _check_ffmpeg_installed():
            click.secho("ffmpeg is not installed or not found in PATH. Skipping compression.", fg='yellow')
            return
        compressed_output = Path(output).with_stem(Path(output).stem + "_compressed")
        cmd = [
            "ffmpeg", "-y", "-i", output,
            "-vcodec", "libx264", "-crf", str(crf),
            "-preset", "slow",
            str(compressed_output)
        ]
        click.secho(f"Compressing {output} to {compressed_output} with CRF={crf}...", fg='yellow')
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            click.secho(f"Compressed video saved as {compressed_output}", fg='green', bold=True)
        else:
            click.secho(f"Compression failed: {result.stderr.decode()}", fg='red')


def parse_timeframe(timeframe):
    start_str, end_str = timeframe.split('-')
    start_h, start_m = map(int, start_str.split(':'))
    end_h, end_m = map(int, end_str.split(':'))
    return (start_h * 60 + start_m, end_h * 60 + end_m)

def is_excluded_time(dt, start_min, end_min):
    mins = dt.hour * 60 + dt.minute
    if start_min <= end_min:
        return start_min <= mins <= end_min
    else:  # overnight
        return mins >= start_min or mins <= end_min

def parse_days(days_str):
    day_map = {d.lower()[:3]: i for i, d in enumerate(calendar.day_name)}
    return set(day_map[d.strip().lower()] for d in days_str.split(',') if d.strip().lower() in day_map)

def is_excluded_day(dt, excluded_days):
    return dt.weekday() in excluded_days

@cli.command('remove-photos')
@click.argument('search_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--exclude-time', default='22:30-04:30', show_default=True, help='Timeframe to exclude (e.g., 22:30-04:30)')
@click.option('--exclude-days', default='sat,sun', show_default=True, help='Days to exclude (comma-separated, e.g., sat,sun)')
@click.option('--restore-removed', is_flag=True, default=False, show_default=True, help='Move previously removed images back before running removal.')
@click.option('--rename', type=str, default=None, help='Rename all files using EXIF date/time and this suffix, e.g. "Own_Name".')
def remove_photos(search_dir, exclude_time, exclude_days, restore_removed, rename):
    search_dir = Path(search_dir)
    removed_dir = search_dir / "removed"
    removed_dir.mkdir(exist_ok=True)

    # Restore previously removed images if requested
    if restore_removed and removed_dir.exists():
        restored = 0
        for f in removed_dir.iterdir():
            if f.is_file():
                dest = search_dir / f.name
                shutil.move(str(f), str(dest))
                restored += 1
        if restored:
            click.secho(f"Restored {restored} images from 'removed' folder.", fg='cyan')
        else:
            click.secho("No images to restore from 'removed' folder.", fg='yellow')

    start_min, end_min = parse_timeframe(exclude_time)
    excluded_days = parse_days(exclude_days)

    files = list_images_recursive(search_dir, ALLOWED_IMAGE_EXTENSIONS)

    if not files:
        click.secho("No image files found.", fg='red')
        return

    # Pre-read EXIF datetimes in parallel
    dts: List[Optional[datetime]] = [None] * len(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8) * 2)) as executor:
        for idx, dt in executor.map(lambda p: (p[0], read_exif_datetime(p[1])), list(enumerate(files))):
            dts[idx] = dt

    total, moved, skipped = 0, 0, 0
    for file_path, dt in zip(files, dts):
        total += 1
        file_path_obj = Path(file_path)
        if dt is None:
            click.secho(f"Skipping '{file_path_obj}' - No EXIF date/time found", fg='yellow')
            skipped += 1
            continue

        # Always rename if requested, regardless of exclusion
        if rename:
            new_name = dt.strftime("%Y-%m-%d-%H-%M-%S") + f"-{rename}{file_path_obj.suffix.lower()}"
            new_path = file_path_obj.parent / new_name
            # Avoid overwriting files
            counter = 1
            while new_path.exists():
                new_name = dt.strftime("%Y-%m-%d-%H-%M-%S") + f"-{rename}-{counter}{file_path_obj.suffix.lower()}"
                new_path = file_path_obj.parent / new_name
                counter += 1
            file_path_obj.rename(new_path)
            file_path_obj = new_path

        if is_excluded_time(dt, start_min, end_min) or is_excluded_day(dt, excluded_days):
            dest = removed_dir / file_path_obj.name
            # Avoid overwriting in removed folder
            counter = 1
            orig_dest = dest
            while dest.exists():
                dest = removed_dir / (orig_dest.stem + f"-{counter}" + orig_dest.suffix)
                counter += 1
            shutil.move(str(file_path_obj), str(dest))
            click.secho(f"Moved '{file_path_obj}' - taken at {dt.strftime('%H:%M')} (excluded)", fg='red')
            moved += 1
        else:
            click.secho(f"Keeping '{file_path_obj}' - taken at {dt.strftime('%H:%M')} (included)", fg='green')

    click.secho("\n=== Summary ===", fg='cyan', bold=True)
    click.secho(f"Files processed: {total}", fg='blue')
    click.secho(f"Files moved: {moved}", fg='red')
    click.secho(f"Files skipped (no EXIF): {skipped}", fg='yellow')
    click.secho(f"Files kept: {total - moved - skipped}", fg='green')

@cli.command('batch-crop')
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('output_folder', type=click.Path())
@click.option('--start-x', type=int, required=True, help='Start X pixel (upper left corner) of crop window.')
@click.option('--start-y', type=int, required=True, help='Start Y pixel (upper left corner) of crop window.')
@click.option('--end-x', type=int, required=True, help='End X pixel (lower right corner) of crop window.')
@click.option('--end-y', type=int, required=True, help='End Y pixel (lower right corner) of crop window.')
@click.option('--rotate', type=float, default=0.0, show_default=True, help='Rotate image by degrees before cropping.')
def batch_crop(input_folder, output_folder, start_x, start_y, end_x, end_y, rotate):
    """
    Batch crop images to a rectangle defined by upper left and lower right corners, preserving EXIF.
    Optionally rotate the image before cropping.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".heic") and f.is_file()]
    if not files:
        click.secho("No image files found.", fg='red')
        return

    def process_image(img_path):
        try:
            with Image.open(img_path) as im:
                # Auto-orient based on EXIF
                im = ImageOps.exif_transpose(im)
                exif = im.getexif()
                # Reset Orientation to 1 after transpose
                if exif is not None and len(exif):
                    exif[274] = 1  # Orientation tag
                    exif_bytes = exif.tobytes()
                else:
                    exif_bytes = None
                if rotate != 0.0:
                    im = im.rotate(-rotate, expand=True, resample=Image.BICUBIC)
                cropped = im.crop((start_x, start_y, end_x, end_y))
                out_path = output_folder / img_path.name
                if exif_bytes:
                    cropped.save(out_path, exif=exif_bytes)
                else:
                    cropped.save(out_path)
            return f"Cropped and saved: {img_path.name}"
        except Exception as e:
            return f"Failed to crop {img_path.name}: {e}"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, files))
    for res in results:
        click.secho(res, fg='green' if res.startswith("Cropped") else 'red')

def write_oriented_image(dt, img_path, out_path):
    """Write image to out_path after applying EXIF Orientation. Preserve other EXIF, set Orientation=1.
    Falls back to copying if anything fails.
    """
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            exif = im.getexif()
            exif_bytes = None
            if exif is not None and len(exif):
                exif[274] = 1  # Orientation
                try:
                    exif_bytes = exif.tobytes()
                except Exception:
                    exif_bytes = None
            if exif_bytes:
                im.save(out_path, exif=exif_bytes)
            else:
                im.save(out_path)
    except Exception:
        shutil.copy2(img_path, out_path)
    return out_path

@cli.command('normalize-intervals')
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('output_folder', type=click.Path())
@click.option('--target-minutes', type=int, default=1, show_default=True, help='Target interval in minutes between frames.')
@click.option('--max-gap-multiplier', type=float, default=3.0, show_default=True,
              help='If the gap between consecutive photos exceeds this multiple of the target interval, do not fill it with duplicated frames.')
def normalize_intervals(input_folder, output_folder, target_minutes, max_gap_multiplier):
    """
    Normalize image intervals by skipping or duplicating frames so all intervals are exactly target_minutes apart.
    If a gap between two real photos exceeds max_gap_multiplier * target_minutes, the gap is left empty (no duplicated frames).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".heic") and f.is_file()]
    if not files:
        click.secho("No image files found.", fg='red')
        return

    # Extract timestamps (parallel) and sort
    dts: List[Optional[datetime]] = [None] * len(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8) * 2)) as executor:
        for idx, dt in executor.map(lambda p: (p[0], read_exif_datetime(str(p[1]))), list(enumerate(files))):
            dts[idx] = dt

    images_with_dt: List[Tuple[datetime, Path]] = []
    for file_path, dt in zip(files, dts):
        if dt is not None:
            images_with_dt.append((dt, file_path))

    images_with_dt.sort()
    if not images_with_dt:
        click.secho("No images with valid EXIF date/time found.", fg='red')
        return

    # Determine the range and incremental start
    first_dt = images_with_dt[0][0]
    last_dt = images_with_dt[-1][0]
    target_delta = target_minutes * 60  # seconds
    threshold_seconds = max_gap_multiplier * target_delta
    existing_last_dt = find_last_normalized_dt(output_folder)
    if existing_last_dt is not None:
        start_dt = max(first_dt, existing_last_dt + timedelta(seconds=target_delta))
    else:
        start_dt = first_dt

    # Precompute which consecutive image pairs constitute large gaps
    large_gap_pairs: List[int] = []  # indices i where gap from i -> i+1 is large
    for i in range(len(images_with_dt) - 1):
        gap_sec = (images_with_dt[i + 1][0] - images_with_dt[i][0]).total_seconds()
        if gap_sec > threshold_seconds:
            large_gap_pairs.append(i)

    # Build normalized timeline starting after existing outputs (incremental)
    normalized_times: List[datetime] = []
    t = start_dt
    while t <= last_dt:
        normalized_times.append(t)
        t = t + timedelta(seconds=target_delta)

    # For each normalized time, find the closest image at or before that time
    result: List[Tuple[datetime, Path, Path]] = []
    img_idx = 0
    skipped_due_to_large_gap = 0
    for norm_time in normalized_times:
        # Advance img_idx to the last image at or before norm_time
        while img_idx + 1 < len(images_with_dt) and images_with_dt[img_idx + 1][0] <= norm_time:
            img_idx += 1
        # If we are between a large gap, skip generating frames until the next real image time
        if img_idx < len(images_with_dt) - 1:
            next_dt = images_with_dt[img_idx + 1][0]
            cur_dt = images_with_dt[img_idx][0]
            gap_sec = (next_dt - cur_dt).total_seconds()
            if gap_sec > threshold_seconds and cur_dt < norm_time < next_dt:
                skipped_due_to_large_gap += 1
                continue
        dt, img_path = images_with_dt[img_idx]
        # Use normalized timeline time for filename, add a counter if needed
        base_name = norm_time.strftime("%Y-%m-%d-%H-%M-%S")
        out_name = f"{base_name}{img_path.suffix.lower()}"
        out_path = output_folder / out_name
        counter = 1
        while out_path.exists():
            out_name = f"{base_name}-{counter}{img_path.suffix.lower()}"
            out_path = output_folder / out_name
            counter += 1
        result.append((dt, img_path, out_path))

    added = 0
    if result:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8) * 2)) as executor:
            list(executor.map(lambda args: write_oriented_image(*args), result))
        added = len(result)

    # Write a state file describing the normalization run
    try:
        import json as _json
        state_data = {
            "target_minutes": target_minutes,
            "max_gap_multiplier": max_gap_multiplier,
            "target_delta_seconds": target_delta,
            "large_gap_threshold_seconds": threshold_seconds,
            "large_gaps_count": len(large_gap_pairs),
            "frames_written": added,
            "frames_skipped_large_gaps": skipped_due_to_large_gap,
            "first_input_dt": first_dt.isoformat(),
            "last_input_dt": last_dt.isoformat(),
            "start_normalized_dt": start_dt.isoformat(),
            "end_normalized_dt": (normalized_times[-1].isoformat() if normalized_times else None),
        }
        state_path = Path(output_folder) / "normalize-intervals.state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            _json.dump(state_data, f, indent=2)
    except Exception:
        pass

    if existing_last_dt is None:
        click.secho(f"Normalized sequence saved to {output_folder} (frames: {added}, skipped due to large gaps: {skipped_due_to_large_gap})", fg='green')
    else:
        click.secho(f"Appended {added} new frames to {output_folder} (skipped due to large gaps: {skipped_due_to_large_gap})", fg='green' if added > 0 else 'yellow')


if __name__ == '__main__':
    cli()