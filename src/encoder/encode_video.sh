#!/bin/bash

INPUT="data/raw_video/test_video.mp4"
OUTPUT_DIR="data/encoded_video"

echo "Starting multi-bitrate encoding..."
echo "Input file: $INPUT"

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    echo "Please create the test video first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# === Quality Level 1: Very Low (240p, 400kbps) ===
echo ""
echo "========================================="
echo "Encoding Quality 1/5: 240p at 400kbps..."
echo "========================================="
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v 400k \
    -maxrate 400k \
    -bufsize 800k \
    -vf scale=426:240 \
    -c:a aac \
    -b:a 64k \
    -keyint_min 48 \
    -g 48 \
    -sc_threshold 0 \
    -preset fast \
    -profile:v baseline \
    -movflags +faststart \
    -y \
    "$OUTPUT_DIR/output_240p.mp4"

echo "240p DONE!"

# === Quality Level 2: Low (360p, 800kbps) ===
echo ""
echo "========================================="
echo "Encoding Quality 2/5: 360p at 800kbps..."
echo "========================================="
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v 800k \
    -maxrate 800k \
    -bufsize 1600k \
    -vf scale=640:360 \
    -c:a aac \
    -b:a 96k \
    -keyint_min 48 \
    -g 48 \
    -sc_threshold 0 \
    -preset fast \
    -movflags +faststart \
    -y \
    "$OUTPUT_DIR/output_360p.mp4"

echo "360p DONE!"

# === Quality Level 3: Medium (480p, 1.5Mbps) ===
echo ""
echo "========================================="
echo "Encoding Quality 3/5: 480p at 1.5Mbps..."
echo "========================================="
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v 1500k \
    -maxrate 1500k \
    -bufsize 3000k \
    -vf scale=854:480 \
    -c:a aac \
    -b:a 128k \
    -keyint_min 48 \
    -g 48 \
    -sc_threshold 0 \
    -preset fast \
    -movflags +faststart \
    -y \
    "$OUTPUT_DIR/output_480p.mp4"

echo "480p DONE!"

# === Quality Level 4: High (720p, 3Mbps) ===
echo ""
echo "========================================="
echo "Encoding Quality 4/5: 720p at 3Mbps..."
echo "========================================="
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v 3000k \
    -maxrate 3000k \
    -bufsize 6000k \
    -vf scale=1280:720 \
    -c:a aac \
    -b:a 128k \
    -keyint_min 48 \
    -g 48 \
    -sc_threshold 0 \
    -preset fast \
    -movflags +faststart \
    -y \
    "$OUTPUT_DIR/output_720p.mp4"

echo "720p DONE!"

# === Quality Level 5: Very High (1080p, 6Mbps) ===
echo ""
echo "========================================="
echo "Encoding Quality 5/5: 1080p at 6Mbps..."
echo "========================================="
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v 6000k \
    -maxrate 6000k \
    -bufsize 12000k \
    -vf scale=1920:1080 \
    -c:a aac \
    -b:a 192k \
    -keyint_min 48 \
    -g 48 \
    -sc_threshold 0 \
    -preset fast \
    -movflags +faststart \
    -y \
    "$OUTPUT_DIR/output_1080p.mp4"

echo "1080p DONE!"

# === Show Results ===
echo ""
echo "========================================="
echo "ALL ENCODING COMPLETE!"
echo "========================================="
echo "Output files:"
ls -lh "$OUTPUT_DIR/"

