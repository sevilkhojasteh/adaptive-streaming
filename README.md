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
