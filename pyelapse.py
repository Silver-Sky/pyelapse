import os
from pathlib import Path

import cv2
import click
import json
import subprocess
import shutil
from datetime import datetime


@click.group()
def cli():
    """PyElapse CLI - Create time-lapse videos from images."""
    pass

@cli.command('create-timelapse')
@click.argument('folder', type=click.Path(exists=True, file_okay=False))
@click.option('--fps', default=24, help='Frames per second for the output video.')
@click.option('--output', default='output.mp4', help='Output video file name.')
def create_timelapse(folder, fps, output):
    # Get sorted list of image files
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not images:
        click.secho('No images found in the folder.', fg='red')
        return

    # Read first image to get frame size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()
    click.secho(f'Time-lapse video saved as {output}', fg='green', bold=True)



def is_night_time(dt):
    mins = dt.hour * 60 + dt.minute
    return mins >= 1350 or mins <= 270  # 22:30–23:59 or 00:00–04:30

def is_weekend(dt):
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday

import calendar

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
def remove_photos(search_dir, exclude_time, exclude_days):
    search_dir = Path(search_dir)
    removed_dir = search_dir / "removed"
    removed_dir.mkdir(exist_ok=True)

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

    cmd = ["exiftool", "-json", "-DateTimeOriginal", "-CreateDate"] + files
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    total, moved, skipped = 0, 0, 0
    for entry in data:
        total += 1
        dt_str = entry.get("DateTimeOriginal") or entry.get("CreateDate")
        file_path = Path(entry["SourceFile"])
        if not dt_str:
            click.secho(f"Skipping '{file_path}' - No EXIF date/time found", fg='yellow')
            skipped += 1
            continue
        try:
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except Exception:
            click.secho(f"Skipping '{file_path}' - Invalid date format", fg='yellow')
            skipped += 1
            continue
        if is_excluded_time(dt, start_min, end_min) or is_excluded_day(dt, excluded_days):
            dest = removed_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            click.secho(f"Moved '{file_path}' - taken at {dt.strftime('%H:%M')} (excluded)", fg='red')
            moved += 1
        else:
            click.secho(f"Keeping '{file_path}' - taken at {dt.strftime('%H:%M')} (included)", fg='green')

    click.secho("\n=== Summary ===", fg='cyan', bold=True)
    click.secho(f"Files processed: {total}", fg='blue')
    click.secho(f"Files moved: {moved}", fg='red')
    click.secho(f"Files skipped (no EXIF): {skipped}", fg='yellow')
    click.secho(f"Files kept: {total - moved - skipped}", fg='green')


if __name__ == '__main__':
    cli()