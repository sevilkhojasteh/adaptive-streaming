# adaptive-streaming

## In order to inspect the video you can use the following structure
# Always first inspect your video:
ffprobe -v quiet -print_format json -show_format -show_streams \
        data/raw_video/test_video.mp4

# Output tells you:
# - Duration
# - Resolution (width, height)
# - Frame rate
# - Current codec
# - Current bitrate
# - Audio codec

# Simpler version:
ffmpeg -i data/raw_video/test_video.mp4

## Create the CLEAN Encoding Script
# Open a text editor to create the script:
nano src/encoder/encode_video.sh

ffmpeg -i "$INPUT" \
    -c:v libx264 \          # Encodes video using the H.264 standard codec
    -b:v 400k \             # Sets target video bitrate to 400 kbps (kilobits per second)
    -maxrate 400k \         # Locks the maximum bitrate so it doesn't spike past 400k
    -bufsize 800k \         # Sets the decoder buffer size to 2x the bitrate for rate-control stability
    -vf scale=426:240 \     # Video Filter: Rescales the frame size down to 426x240 pixels (240p)
    -c:a aac \              # Encodes the audio track into AAC format
    -b:a 64k \              # Sets audio bitrate to 64 kbps (lower quality for lower video tier)
    -keyint_min 48 \        # Forces a minimum interval of 48 frames between keyframes (I-frames)
    -g 48 \                 # Sets the GOP (Group of Pictures) size to 48 frames. At 24fps, this equals exactly 2 seconds.
    -sc_threshold 0 \       # Disables automatic scene-change detection to lock keyframes strictly every 2 seconds
    -preset fast \          # Balances encoding speed and compression efficiency (options range from ultrafast to veryslow)
    -profile:v baseline \   # Uses H.264 Baseline profile for maximum compatibility with older devices
    -movflags +faststart \  # Moves metadata to the start of the file for quick web loading
    "$OUTPUT_DIR/output_240p.mp4"


## Create and run the DASH packaging script
Because inline comments (like \ # comment) break bash scripts, let's create a clean script file.


# Open a file for the DASH script:
nano src/encoder/create_dash.sh
Copy and paste this clean code into nano:


#!/bin/bash

OUTPUT_DIR="data/encoded_video"

echo "========================================="
echo "Creating DASH Segments and Manifest..."
echo "========================================="

# Check if encoded files exist
if [ ! -f "$OUTPUT_DIR/output_1080p.mp4" ]; then
    echo "ERROR: Encoded videos not found in $OUTPUT_DIR"
    echo "Run encode_video.py first!"
    exit 1
fi

# Run MP4Box to package files into DASH
MP4Box -dash 4000 \
       -frag 4000 \
       -rap \
       -segment-name 'seg_$RepresentationID$_' \
       -out "$OUTPUT_DIR/manifest.mpd" \
       "$OUTPUT_DIR/output_240p.mp4" \
       "$OUTPUT_DIR/output_360p.mp4" \
       "$OUTPUT_DIR/output_480p.mp4" \
       "$OUTPUT_DIR/output_720p.mp4" \
       "$OUTPUT_DIR/output_1080p.mp4"

echo ""
echo "========================================="
echo "DASH PACKAGING COMPLETE!"
echo "========================================="
ls -lh "$OUTPUT_DIR/"
Save and exit nano: Press Ctrl+X, then Y, then Enter.
