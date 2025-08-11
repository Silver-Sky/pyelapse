# PyElapse

A Python script for creating time-lapse videos from images.

If you have found this, you probably have the same issue I had: There are plenty of tools to manipulate images, but some of them lack bulk processing capabilities. This script is meant to fill that gap for my use case.
I wanted to create a time-lapse video of my construction site and needed to remove night photos, weekends, crop images, and normalize the intervals of the images.
It is far from perfect, but it works for me. If you have any suggestions or improvements, feel free to open an issue or a pull request.

Example usage:

```bash
pyelapse remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun
pyelapse create-timelapse /path/to_image_folder/ --output path/time-elapse-video.mp4 --fps 30
```

## install

pip install pyelapse

## Remove Photos

You can try out different removal settings by restoring previously removed images before running the removal:

```bash
pyelapse remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun --restore-removed
```

- The `--restore-removed` flag moves all images from the `removed` folder back to the main folder before applying the removal logic.
- You can also rename the kept files using the EXIF date/time and a custom suffix with `--rename`, e.g.:

```bash
pyelapse remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun --rename Own_Name
```
This will produce files like `2024-06-07-14-30-00-Own_Name.jpg`.

## Batch Crop Images

Crop all images in a folder to a rectangle, specifying the upper left and lower right corners, and preserving EXIF data:

```bash
pyelapse batch-crop /path/to_input_folder/ /path/to_output_folder/ --start-x 100 --start-y 50 --end-x 700 --end-y 950
```

- The crop window is defined by the upper left (`--start-x`, `--start-y`) and lower right (`--end-x`, `--end-y`) corners.
- EXIF information is preserved in the output images.
- You can optionally rotate the image before cropping using `--rotate`, e.g.:

```bash
python pyelapse.py batch-crop /input /output --start-x 100 --start-y 50 --end-x 700 --end-y 950 --rotate 4.14
```

- You can also rename the output files using the EXIF date/time and a custom suffix with `--rename`, e.g.:

```bash
python pyelapse.py batch-crop /input /output --start-x 100 --start-y 50 --end-x 700 --end-y 950 --rename Own_Name
```
This will produce files like `2024-06-07-14-30-00-Own_Name.jpg`.

## Normalize Image Intervals

If your images have different time intervals (e.g., some 3 minutes apart, some 1 minute apart), you can normalize them so the video plays at a constant speed:

```bash
python pyelapse.py normalize-intervals /path/to_input_folder/ /path/to_output_folder/ --target-minutes 1
```

- This will duplicate or skip frames as needed so all intervals match the target (e.g., 1 minute).
- Use the output folder as input for your time-lapse creation.

## Create Timelapse Video

You can compress the output `.mov` file using ffmpeg and H.264 by specifying the `--crf` option (lower values mean higher quality, typical range 18-28):

```bash
python pyelapse.py create-timelapse /path/to_image_folder/ --output output.mov --fps 24 --crf 23
```

This will create `output.mov` and then compress it to `output_compressed.mov` using ffmpeg.

### Timestamps on frames

To overlay the EXIF date/time in the lower-right corner of each frame while creating the timelapse, add `--timestamp`:

```bash
python pyelapse.py create-timelapse /path/to_image_folder/ --output output.mp4 --fps 24 --timestamp
```

- If EXIF `DateTimeOriginal` is present, that timestamp is used.
- If EXIF is missing, the script falls back to the filename (if it contains digits), otherwise to the current date/time at build.