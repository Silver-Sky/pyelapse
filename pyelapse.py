import os
import calendar

import cv2
import click
import json
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import exifread

from datetime import datetime, timedelta
import concurrent.futures


@click.group()
def cli():
    """PyElapse CLI - Create time-lapse videos from images."""
    pass

@cli.command('create-timelapse')
@click.argument('folder', type=click.Path(exists=True, file_okay=False))
@click.option('--fps', default=24, help='Frames per second for the output video.')
@click.option('--output', default='output.mov', help='Output video file name (H.264 in .mov container).')
@click.option('--crf', type=int, default=None, help='Compress output using ffmpeg with given CRF (lower is better, 18-28).')
def create_timelapse(folder, fps, output, crf):
    """
    Create a time-lapse video from images in a folder.
    :param folder: Path to the folder containing images.
    :param fps: frames per second for the output video.
    :param output: filename for the output video.
    :return: None
    """
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not images:
        click.secho('No images found in the folder.', fg='red')
        return

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Select codec based on file extension
    ext = os.path.splitext(output)[1].lower()
    if ext == ".mov":
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 for .mov
        codec_name = "H.264 (avc1) in .mov"
    elif ext in [".mp4", ".m4v"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264/MP4
        codec_name = "H.264 (mp4v) in .mp4/.m4v"
    elif ext == ".mkv":
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 for .mkv
        codec_name = "H.264 (X264) in .mkv"
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to mp4v
        codec_name = "H.264 (mp4v) default"

    click.secho("=== Timelapse Creation Details ===", fg='cyan', bold=True)
    click.secho(f"Images used: {len(images)}", fg='blue')
    click.secho(f"Resolution: {width}x{height}", fg='blue')
    click.secho(f"FPS: {fps}", fg='blue')
    click.secho(f"Codec: {codec_name}", fg='blue')
    click.secho(f"Output file: {output}", fg='blue')
    click.secho(f"First image: {os.path.basename(images[0])}", fg='blue')
    click.secho(f"Last image: {os.path.basename(images[-1])}", fg='blue')
    click.secho("=" * 32, fg='cyan')

    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    with click.progressbar(images, label="Creating timelapse") as bar:
        for img_path in bar:
            img = cv2.imread(img_path)
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            out.write(img)

    out.release()
    click.secho(f"Time-lapse video saved as {output}", fg='green', bold=True)

    # Compress with ffmpeg if requested
    if crf is not None:
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
@click.option('--exclude-time', default='22:30-04:30', help='Timeframe to exclude (e.g., 22:30-04:30)')
@click.option('--exclude-days', default='sat,sun', help='Days to exclude (comma-separated, e.g., sat,sun)')
@click.option('--restore-removed', is_flag=True, default=False, help='Move previously removed images back before running removal.')
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

    exts = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw")
    files = []
    for root, _, filenames in os.walk(search_dir):
        root_path = Path(root)
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(str(root_path / f))

    if not files:
        click.secho("No image files found.", fg='red')
        return

    total, moved, skipped = 0, 0, 0
    for file_path in files:
        total += 1
        dt_str = None
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
                dt_str = str(tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime"))
        except Exception:
            dt_str = None

        file_path_obj = Path(file_path)
        if not dt_str or dt_str == "None":
            click.secho(f"Skipping '{file_path_obj}' - No EXIF date/time found", fg='yellow')
            skipped += 1
            continue
        try:
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except Exception:
            click.secho(f"Skipping '{file_path_obj}' - Invalid date format", fg='yellow')
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

    exts = (".jpg", ".jpeg", ".png", ".tiff")
    files = [f for f in input_folder.iterdir() if f.suffix.lower() in exts and f.is_file()]
    if not files:
        click.secho("No image files found.", fg='red')
        return

    def process_image(img_path):
        try:
            with Image.open(img_path) as im:
                exif = im.info.get('exif')
                if rotate != 0.0:
                    im = im.rotate(-rotate, expand=True, resample=Image.BICUBIC)
                cropped = im.crop((start_x, start_y, end_x, end_y))
                out_path = output_folder / img_path.name
                if exif:
                    cropped.save(out_path, exif=exif)
                else:
                    cropped.save(out_path)
            return f"Cropped and saved: {img_path.name}"
        except Exception as e:
            return f"Failed to crop {img_path.name}: {e}"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, files))
    for res in results:
        click.secho(res, fg='green' if res.startswith("Cropped") else 'red')

def copy_image(dt, img_path, out_path):
    shutil.copy2(img_path, out_path)
    return out_path

@cli.command('normalize-intervals')
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('output_folder', type=click.Path())
@click.option('--target-minutes', type=int, default=1, show_default=True, help='Target interval in minutes between frames.')
def normalize_intervals(input_folder, output_folder, target_minutes):
    """
    Normalize image intervals by skipping or duplicating frames so all intervals are exactly target_minutes apart.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".tiff")
    files = [f for f in input_folder.iterdir() if f.suffix.lower() in exts and f.is_file()]
    if not files:
        click.secho("No image files found.", fg='red')
        return

    # Extract timestamps and sort
    images_with_dt = []
    for img_path in files:
        try:
            with open(img_path, 'rb') as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
                dt_str = str(tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime"))
            if dt_str and dt_str != "None":
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                images_with_dt.append((dt, img_path))
        except Exception:
            continue

    images_with_dt.sort()
    if not images_with_dt:
        click.secho("No images with valid EXIF date/time found.", fg='red')
        return

    # Determine the range
    first_dt = images_with_dt[0][0]
    last_dt = images_with_dt[-1][0]
    target_delta = target_minutes * 60  # seconds

    # Build normalized timeline
    normalized_times = []
    t = first_dt
    while t <= last_dt:
        normalized_times.append(t)
        t = t + timedelta(seconds=target_delta)

    # For each normalized time, find the closest image at or before that time
    result = []
    img_idx = 0
    for norm_time in normalized_times:
        # Advance img_idx to the last image at or before norm_time
        while img_idx + 1 < len(images_with_dt) and images_with_dt[img_idx + 1][0] <= norm_time:
            img_idx += 1
        dt, img_path = images_with_dt[img_idx]
        # Use EXIF date/time for filename, add a counter if needed
        base_name = norm_time.strftime("%Y-%m-%d-%H-%M-%S")
        out_name = f"{base_name}{img_path.suffix.lower()}"
        out_path = output_folder / out_name
        counter = 1
        while out_path.exists():
            out_name = f"{base_name}-{counter}{img_path.suffix.lower()}"
            out_path = output_folder / out_name
            counter += 1
        result.append((dt, img_path, out_path))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(lambda args: copy_image(*args), result))

    click.secho(f"Normalized sequence saved to {output_folder}", fg='green')


if __name__ == '__main__':
    cli()