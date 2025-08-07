#!/usr/bin/env bash
source .venv/bin/activate

# Define intervals and folders
intervals=(3 15)
folders=("1_Aushub_full" "2_Keller" "3_Keller_ferien" "4_Kanalisation")
# folders=("4_Kanalisation")

for interval in "${intervals[@]}"; do
    norm_folder="/Volumes/T7/time-elapse-data/images/Time-Elapse-Normalized-${interval}min/"
    rm -rf "${norm_folder}"
    mkdir -p "${norm_folder}"
    for folder in "${folders[@]}"; do
        python pyelapse.py normalize-intervals "/Volumes/T7/time-elapse-data/images/${folder}/" "$norm_folder" --target-minutes "$interval"
    done

    # Remove night photos and weekends
    python pyelapse.py remove-photos "$norm_folder" --exclude-time 22:30-04:30 --exclude-days sat,sun --restore-removed
    python pyelapse.py create-timelapse "$norm_folder" --output "/Volumes/T7/time-elapse-data/videos/Time_Elapse_fps30_${interval}min.mov" --fps 30

    # Only Workday photos
    python pyelapse.py remove-photos "$norm_folder" --exclude-time 18:30-06:30 --exclude-days sat,sun --restore-removed
    python pyelapse.py create-timelapse "$norm_folder" --output "/Volumes/T7/time-elapse-data/videos/Time_Elapse_only_day_fps30_${interval}min.mov" --fps 30
done
