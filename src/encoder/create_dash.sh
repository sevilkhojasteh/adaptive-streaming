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
