# PyElapse

A Python script for creating time-lapse videos from images.

Example usage:

```bash
source .venv/bin/activate
python pyelapse.py remove-photos /path/to_image_folder/ --exclude-time 22:00-06:00 --exclude-days sat,sun
python pyelapse.py create-video /path/to_image_folder/ --output path/time-elapse-video.mp4 --fps 30
```