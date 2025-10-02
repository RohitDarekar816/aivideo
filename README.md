# Video Processing Tool - Docker Setup

This project combines video clips with subtitles, audio tracks, watermarks, and transitions to create professional videos. The tool is containerized with Docker for easy deployment and sharing across teams.

## Features

- 🎬 **Video Clip Combination**: Merge multiple video clips with smooth transitions
- 📝 **Subtitle Overlay**: Support for SRT and custom timestamp formats (with automatic text capitalization)
- 🎵 **Audio Replacement**: Replace video audio with external WAV files
- 🖼️ **Watermarking**: Add YouTube-style watermarks
- ✨ **Transitions**: Fadeblack, crossfade, or no transition options
- 🎨 **Customizable Text**: Font size, family, color, and stroke options

## Prerequisites

- Docker
- Docker Compose

## Quick Start

### 1. Prepare Your Files

Create the following directory structure:
```
project-folder/
├── clips/                    # Put your video clips here
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
├── analyzed_filepathX75.wav  # Your audio file
├── transcript_output.txt     # Your transcript/subtitles
├── watermark.jpeg           # Your watermark (optional)
└── output/                  # Generated videos will be here
```

### 2. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Run in test mode (60 seconds preview)
docker-compose run --rm video-processor python create_video.py --mode test

# Run full production render
docker-compose run --rm video-processor python create_video.py --mode prod
```

### 3. Alternative Docker Commands

```bash
# Build the image manually
docker build -t video-processor .

# Run with custom parameters
docker run --rm -v "$(pwd)/clips:/app/clips" \
                -v "$(pwd)/output:/app/output" \
                -v "$(pwd)/analyzed_filepathX75.wav:/app/analyzed_filepathX75.wav:ro" \
                -v "$(pwd)/transcript_output.txt:/app/transcript_output.txt:ro" \
                -v "$(pwd)/watermark.jpeg:/app/watermark.jpeg:ro" \
                video-processor python create_video.py --mode test
```

## Usage Examples

### Basic Usage
```bash
# Test mode with 30-second preview
docker-compose run --rm video-processor python create_video.py --mode test --test-duration 30

# Production mode with custom output name
docker-compose run --rm video-processor python create_video.py --output my_video.mp4
```

### Advanced Options
```bash
# Custom text styling
docker-compose run --rm video-processor python create_video.py \
  --font-family "Impact" \
  --font-color "yellow" \
  --stroke-color "red" \
  --font-size 60

# Different transitions
docker-compose run --rm video-processor python create_video.py \
  --transition crossfade \
  --transition-duration 1.0

# No watermark
docker-compose run --rm video-processor python create_video.py --no-watermark
```

## Transcript Format Examples

The tool supports multiple transcript formats:

### SRT Format
```
1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,000
This is a test
```

### Line-based with Ranges
```
[00:00:01.000 - 00:00:03.500] Hello world
00:00:04.000 --> 00:00:06.000 This is a test
```

### Single Timestamps
```
00:00:01.000 Hello world
00:00:04.000 This is a test
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--clips-dir` | `clips` | Directory containing video clips |
| `--audio` | `analyzed_filepathX75.wav` | Audio file path |
| `--transcript` | `transcript_output.txt` | Transcript file path |
| `--output` | `output.mp4` | Output video path |
| `--mode` | `prod` | Mode: `prod` or `test` |
| `--test-duration` | `60.0` | Test mode duration in seconds |
| `--transition` | `fadeblack` | Transition: `none`, `fadeblack`, `crossfade` |
| `--font-family` | Auto | Font family (e.g., 'Arial', 'Impact') |
| `--font-color` | `white` | Font color |
| `--stroke-color` | `black` | Outline color |
| `--no-watermark` | False | Disable watermark |

## Troubleshooting

### Permission Issues
If you encounter permission issues with output files:
```bash
# On Linux/Mac, fix permissions
sudo chown -R $USER:$USER output/

# On Windows (PowerShell as Admin)
takeown /f output /r /d y
```

### Missing Files
Make sure all required files exist:
- Video clips in `./clips/` directory
- Audio file: `analyzed_filepathX75.wav`
- Transcript file: `transcript_output.txt`

### Font Issues
The container includes common fonts. If specific fonts aren't available, it will fall back to system defaults.

## Development

To modify the script and rebuild:
```bash
# Edit create_video.py
# Then rebuild the image
docker-compose build --no-cache
```

## Notes

- All subtitle text is automatically converted to UPPERCASE
- The container runs on Linux, so ImageMagick paths are automatically adjusted
- Output videos use H.264 codec with AAC audio for maximum compatibility
- Test mode is recommended for quick iterations and debugging