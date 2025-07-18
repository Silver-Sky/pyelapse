# PyElapse

A Python script for creating time-lapse videos from images.

Example usage:

```bash
source .venv/bin/activate
python pyelapse.py remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun
python pyelapse.py create-video /path/to_image_folder/ --output path/time-elapse-video.mp4 --fps 30
```

## Remove Photos

You can try out different removal settings by restoring previously removed images before running the removal:

```bash
python pyelapse.py remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun --restore-removed
```

- The `--restore-removed` flag moves all images from the `removed` folder back to the main folder before applying the removal logic.
- You can also rename the kept files using the EXIF date/time and a custom suffix with `--rename`, e.g.:

```bash
python pyelapse.py remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun --rename Own_Name
```
This will produce files like `2024-06-07-14-30-00-Own_Name.jpg`.

## Batch Crop Images

Crop all images in a folder to a rectangle, specifying the upper left and lower right corners, and preserving EXIF data:

```bash
python pyelapse.py batch-crop /path/to_input_folder/ /path/to_output_folder/ --start-x 100 --start-y 50 --end-x 700 --end-y 950
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
