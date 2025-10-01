#!/usr/bin/env python3
"""
create_video.py

Combines video clips from a folder, overlays customizable subtitles from a transcript
(with timestamps), applies an external WAV audio track, adds watermarks, and supports
transitions between clips. Outputs an MP4 video.

Defaults:
- Clips directory: ./clips
- Audio file: ./analyzed_filepathX75.wav
- Transcript file: ./transcript_output.txt
- Output file: ./output.mp4

Transcript formats supported (auto-detected):
1) SRT style blocks:
   1
   00:00:01,000 --> 00:00:03,000
   Your text here

2) Line-based with explicit start-end:
   [00:00:01.000 - 00:00:03.000] Your text
   00:00:01.000 --> 00:00:03.000 Your text
   00:01:02 - 00:01:05 Another text
   62.0 - 65.0 Another text

3) Line-based with single start time (duration inferred until next cue or a default):
   00:00:01.250 Your text here
   00:00:04.000 Next text here

If only start times are provided, a default duration of 2.5s is used, or until the next cue.

Requires: moviepy (will pull imageio-ffmpeg), pillow.
Note: If a specified font is not found, it will fall back to system defaults.

New Features:
- Customizable text formatting with Google Fonts support (font size, family, color, bold, stroke)
- Automatic watermark support (uses watermark.jpeg by default, YouTube-style positioning)
- Transition effects between clips (none, fadeblack, crossfade)
- Google Fonts integration (automatically downloads and caches fonts)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
# Configure ImageMagick path via environment variable
import os
import requests
import zipfile
import json
from urllib.parse import quote
os.environ['IMAGEMAGICK_BINARY'] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
    AudioFileClip,
    ImageClip,
    ColorClip,
)
from moviepy.audio.fx.all import audio_loop
from moviepy.video.fx.all import fadein, fadeout


VideoSegment = Tuple[float, float, str]  # (start, end, text)

# Google Fonts support
GOOGLE_FONTS_API_KEY = os.environ.get('GOOGLE_FONTS_API_KEY')  # Optional, for rate limiting
FONTS_CACHE_DIR = Path.home() / '.video_fonts_cache'
GOOGLE_FONTS_API_URL = 'https://www.googleapis.com/webfonts/v1/webfonts'

def ensure_fonts_cache_dir():
    """Ensure the fonts cache directory exists."""
    FONTS_CACHE_DIR.mkdir(exist_ok=True)
    return FONTS_CACHE_DIR

def get_popular_google_fonts():
    """Return a curated list of popular Google Fonts."""
    return [
        'Roboto', 'Open Sans', 'Lato', 'Montserrat', 'Source Sans Pro',
        'Raleway', 'Poppins', 'Oswald', 'Inter', 'Playfair Display',
        'Merriweather', 'Nunito Sans', 'Ubuntu', 'Libre Franklin',
        'Work Sans', 'Fira Sans', 'PT Sans', 'Crimson Text',
        'Libre Baskerville', 'Cormorant Garamond', 'Quicksand',
        'Dancing Script', 'Pacifico', 'Lobster', 'Bebas Neue',
        'Anton', 'Righteous', 'Fredoka One', 'Comfortaa',
        'Permanent Marker', 'Satisfy', 'Great Vibes', 'Kalam',
        'Caveat', 'Indie Flower', 'Shadows Into Light', 'Amatic SC',
        'Press Start 2P', 'Orbitron', 'Audiowide', 'Black Ops One'
    ]

def get_google_fonts_list():
    """Get list of available Google Fonts."""
    try:
        cache_file = FONTS_CACHE_DIR / 'google_fonts_list.json'
        
        # Check if we have a cached list (refresh daily)
        if cache_file.exists():
            import time
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Try to fetch from Google Fonts API
        params = {'sort': 'popularity'}
        if GOOGLE_FONTS_API_KEY:
            params['key'] = GOOGLE_FONTS_API_KEY
        
        try:
            response = requests.get(GOOGLE_FONTS_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                fonts_data = response.json()
                # Cache the response
                ensure_fonts_cache_dir()
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(fonts_data, f)
                return fonts_data
        except:
            pass
        
        # Fallback to curated list
        print("Using curated popular fonts list...")
        popular_fonts = get_popular_google_fonts()
        fallback_data = {
            'items': [{'family': font, 'category': 'sans-serif'} for font in popular_fonts]
        }
        return fallback_data
        
    except Exception as e:
        print(f"Warning: Failed to get Google Fonts list: {e}")
        # Return curated list as last resort
        popular_fonts = get_popular_google_fonts()
        return {
            'items': [{'family': font, 'category': 'sans-serif'} for font in popular_fonts]
        }

def download_google_font(font_family: str, font_weight: str = 'regular'):
    """Download a Google Font and return the local font file path."""
    try:
        ensure_fonts_cache_dir()
        
        # Sanitize font name for filename
        safe_name = re.sub(r'[^a-zA-Z0-9\-_]', '_', font_family.lower())
        font_dir = FONTS_CACHE_DIR / safe_name
        font_file = font_dir / f"{safe_name}_{font_weight}.ttf"
        
        # Return if already downloaded
        if font_file.exists():
            return str(font_file)
        
        print(f"Downloading Google Font: {font_family} ({font_weight})...")
        
        # Try direct Google Fonts CSS API approach
        try:
            # Format font name for URL
            font_url_name = font_family.replace(' ', '+')
            weight_param = ':wght@400' if font_weight == 'regular' else f':wght@{font_weight}'
            
            # Get CSS from Google Fonts
            css_url = f"https://fonts.googleapis.com/css2?family={font_url_name}{weight_param}&display=swap"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            css_response = requests.get(css_url, headers=headers, timeout=10)
            if css_response.status_code == 200:
                css_content = css_response.text
                
                # Extract font URLs from CSS (try different formats)
                font_urls = []
                # Try TTF first
                font_urls.extend(re.findall(r'url\((https://[^)]+\.ttf)\)', css_content))
                # Try WOFF2 as fallback
                if not font_urls:
                    font_urls.extend(re.findall(r'url\((https://[^)]+\.woff2)\)', css_content))
                # Try any font file
                if not font_urls:
                    font_urls.extend(re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', css_content))
                
                if font_urls:
                    font_url = font_urls[0]
                    file_ext = 'ttf' if font_url.endswith('.ttf') else 'woff2'
                    
                    # Update the local filename
                    font_file = font_dir / f"{safe_name}_{font_weight}.{file_ext}"
                    
                    # Download the font file
                    font_response = requests.get(font_url, headers=headers, timeout=30)
                    if font_response.status_code == 200:
                        font_dir.mkdir(exist_ok=True)
                        with open(font_file, 'wb') as f:
                            f.write(font_response.content)
                        print(f"Downloaded {file_ext.upper()} font to: {font_file}")
                        return str(font_file)
                else:
                    print(f"Could not find font URL in CSS for {font_family}")
                    # Debug: show part of CSS content
                    print(f"CSS content preview: {css_content[:200]}...")
            else:
                print(f"Failed to get CSS for {font_family}: HTTP {css_response.status_code}")
        except Exception as e:
            print(f"Direct download failed: {e}")
        
        # Fallback: Try the API approach if available
        fonts_data = get_google_fonts_list()
        if fonts_data:
            # Find the font in the list
            font_info = None
            for font in fonts_data.get('items', []):
                if font['family'].lower() == font_family.lower():
                    font_info = font
                    break
            
            if font_info and 'files' in font_info:
                files = font_info['files']
                if font_weight in files:
                    font_url = files[font_weight]
                    response = requests.get(font_url, timeout=30)
                    if response.status_code == 200:
                        font_dir.mkdir(exist_ok=True)
                        with open(font_file, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded font via API to: {font_file}")
                        return str(font_file)
        
        print(f"Could not download font '{font_family}'")
        return None
            
    except Exception as e:
        print(f"Error downloading Google Font '{font_family}': {e}")
        return None

def get_font_path(font_family: str, font_bold: bool = False) -> Optional[str]:
    """Get the path to a font file, downloading from Google Fonts if needed."""
    if not font_family:
        return None
    
    # Check if it's a local font file path
    if os.path.exists(font_family):
        return font_family
    
    # Try system fonts first (common ones)
    system_fonts = {
        'arial': 'C:\\Windows\\Fonts\\arial.ttf',
        'times': 'C:\\Windows\\Fonts\\times.ttf',
        'calibri': 'C:\\Windows\\Fonts\\calibri.ttf',
        'segoe ui': 'C:\\Windows\\Fonts\\segoeui.ttf',
    }
    
    if font_family.lower() in system_fonts:
        system_path = system_fonts[font_family.lower()]
        if os.path.exists(system_path):
            return system_path
    
    # Try to download from Google Fonts
    weight = '700' if font_bold else 'regular'
    return download_google_font(font_family, weight)


def time_to_seconds(s: str) -> float:
    """Convert a time string to seconds. Supports:
    - HH:MM:SS.mmm | HH:MM:SS,mmm | HH:MM:SS
    - MM:SS.mmm    | MM:SS,mmm    | MM:SS
    - SS.mmm or integer seconds
    """
    s = s.strip()
    # Replace comma with dot for milliseconds
    s = s.replace(",", ".")

    # If it's pure number (possibly float)
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)

    # Split by ':'
    parts = s.split(":")
    parts = [p.strip() for p in parts]
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    elif len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + float(ss)
    elif len(parts) == 1:
        return float(parts[0])
    else:
        raise ValueError(f"Unrecognized time format: {s}")


def parse_srt(text: str) -> List[VideoSegment]:
    segments: List[VideoSegment] = []
    # Split blocks by blank lines
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    time_line_re = re.compile(r"^(?P<s>\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<e>\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*$")
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # Optional index line at the top
        if len(lines) >= 2 and time_line_re.match(lines[1]):
            time_line = lines[1]
            content_lines = lines[2:]
        elif time_line_re.match(lines[0]):
            time_line = lines[0]
            content_lines = lines[1:]
        else:
            # Not an SRT block
            continue
        m = time_line_re.match(time_line)
        if not m:
            continue
        start_s = time_to_seconds(m.group("s"))
        end_s = time_to_seconds(m.group("e"))
        text_content = "\n".join(content_lines).strip()
        if text_content:
            segments.append((start_s, end_s, text_content))
    return segments


def parse_line_based(text: str, default_duration: float = 2.5) -> List[VideoSegment]:
    segments: List[VideoSegment] = []
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]

    # Patterns:
    # 1) [start - end] text  OR start --> end text  OR start - end text
    pattern_range = re.compile(
        r"^\s*\[?\s*(?P<start>[^\]-]+?)\s*(?:-|–|—|to|-->|–>|->|>+|→)\s*(?P<end>[^\]]+?)\s*\]?\s+(?P<text>.+)$",
        re.IGNORECASE,
    )
    # 2) start text (only start time)
    pattern_single = re.compile(r"^\s*(?P<start>\S+)\s+(?P<text>.+)$")

    # Collect single-timestamp entries first, then we can infer end from next start
    temp_single: List[Tuple[float, str]] = []

    for ln in lines:
        m = pattern_range.match(ln)
        if m:
            try:
                s = time_to_seconds(m.group("start"))
                e = time_to_seconds(m.group("end"))
            except Exception:
                continue
            txt = m.group("text").strip()
            if txt and e > s:
                segments.append((s, e, txt))
            continue

        m2 = pattern_single.match(ln)
        if m2:
            try:
                s = time_to_seconds(m2.group("start"))
            except Exception:
                continue
            txt = m2.group("text").strip()
            if txt:
                temp_single.append((s, txt))
            continue

    # Sort and convert single timestamps to ranges
    temp_single.sort(key=lambda x: x[0])
    for idx, (s, txt) in enumerate(temp_single):
        if idx + 1 < len(temp_single):
            e = temp_single[idx + 1][0]
            # Ensure at least a small positive duration
            if e <= s:
                e = s + default_duration
        else:
            e = s + default_duration
        segments.append((s, e, txt))

    # Deduplicate and sort
    segments = [seg for seg in segments if seg[1] > seg[0]]
    segments.sort(key=lambda x: x[0])
    return segments


def parse_transcript_file(path: Path) -> List[VideoSegment]:
    text = path.read_text(encoding="utf-8", errors="replace")

    # Try SRT first
    srt_segments = parse_srt(text)
    if srt_segments:
        return srt_segments

    # Fallback to line-based parsing
    return parse_line_based(text)


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def gather_clips(clips_dir: Path) -> List[VideoFileClip]:
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    files = [p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No video files found in {clips_dir}")

    files.sort(key=lambda p: natural_sort_key(p.name))
    clips: List[VideoFileClip] = []
    for f in files:
        clips.append(VideoFileClip(str(f)))
    return clips


def choose_font_candidates() -> List[str]:
    # A list of commonly available bold fonts on Windows/macOS/Linux.
    return [
        "Arial Black",
        "Impact",
        "Segoe UI Bold",
        "Arial-BoldMT",
        "Arial-Bold",
        "Helvetica-Bold",
        "Verdana Bold",
        "LiberationSans-Bold",
        "DejaVuSans-Bold",
        "DejaVu Sans Bold",
    ]


def make_subtitle_clip(
    txt: str, 
    start: float, 
    end: float, 
    base_w: int, 
    base_h: int,
    font_size: Optional[int] = None,
    font_family: Optional[str] = None,
    font_color: str = "white",
    font_bold: bool = True,
    stroke_color: str = "black",
    stroke_width: Optional[int] = None
) -> Optional[TextClip]:
    duration = max(0.01, end - start)
    fontsize = font_size or max(24, int(base_h * 0.08))  # ~8% of height or custom
    stroke_w = stroke_width or max(2, int(base_h * 0.004))
    max_width = int(base_w * 0.9)

    # Try to get the specified font (including Google Fonts)
    last_error: Optional[Exception] = None
    font_file_path = None
    
    if font_family:
        print(f"Attempting to use font: {font_family}")
        font_file_path = get_font_path(font_family, font_bold)
        
        if font_file_path:
            try:
                print(f"Using font file: {font_file_path}")
                clip = TextClip(
                    txt,
                    fontsize=fontsize,
                    color=font_color,
                    font=font_file_path,
                    method="caption",
                    size=(max_width, None),  # wrap if needed
                    align="center",
                    stroke_color=stroke_color,
                    stroke_width=stroke_w,
                ).set_position("center").set_start(start).set_duration(duration)
                return clip
            except Exception as e:
                last_error = e
                print(f"Failed to use font file {font_file_path}: {e}")
    
    # Fallback to system fonts
    fonts_to_try = []
    if font_bold:
        fonts_to_try.extend(choose_font_candidates())
    else:
        fonts_to_try.extend([
            "Arial", "Helvetica", "Verdana", "Segoe UI", 
            "DejaVu Sans", "Liberation Sans", "Calibri"
        ])
    
    for font in fonts_to_try:
        try:
            clip = TextClip(
                txt,
                fontsize=fontsize,
                color=font_color,
                font=font,
                method="caption",
                size=(max_width, None),  # wrap if needed
                align="center",
                stroke_color=stroke_color,
                stroke_width=stroke_w,
            ).set_position("center").set_start(start).set_duration(duration)
            return clip
        except Exception as e:
            last_error = e
            continue

    # Fallback without specifying font
    try:
        clip = TextClip(
            txt,
            fontsize=fontsize,
            color=font_color,
            method="caption",
            size=(max_width, None),
            align="center",
            stroke_color=stroke_color,
            stroke_width=stroke_w,
        ).set_position("center").set_start(start).set_duration(duration)
        return clip
    except Exception:
        # If even fallback fails, report but continue without this cue
        print(f"Warning: Failed to render subtitle for '{txt[:40]}...' due to: {last_error}")
        return None


def fit_audio_to_duration(audio: AudioFileClip, target_duration: float) -> AudioFileClip:
    # If audio is longer, trim; if shorter, loop it to fit
    if audio.duration >= target_duration:
        return audio.subclip(0, target_duration)
    else:
        return audio_loop(audio, duration=target_duration)


def apply_transitions(clips: List[VideoFileClip], transition: str, duration: float) -> VideoFileClip:
    """Apply transitions between video clips."""
    if len(clips) <= 1 or transition == "none" or duration <= 0:
        return concatenate_videoclips(clips, method="compose")
    
    if transition == "crossfade":
        # Apply crossfade by overlapping clips
        clips_faded = [clips[0]]  # First clip unchanged
        for c in clips[1:]:
            c_faded = c.crossfadein(duration)
            clips_faded.append(c_faded)
        return concatenate_videoclips(clips_faded, method="compose", padding=-duration)
    
    elif transition == "fadeblack":
        # Add black fade between clips
        seq = []
        w, h = clips[0].w, clips[0].h
        black = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)
        
        for i, c in enumerate(clips):
            ci = c
            # Fade out all clips except the last
            if i < len(clips) - 1:
                ci = fadeout(ci, duration)
            # Fade in all clips except the first
            if i > 0:
                ci = fadein(ci, duration)
            seq.append(ci)
            # Add black transition between clips (except after the last)
            if i < len(clips) - 1:
                seq.append(black)
        
        return concatenate_videoclips(seq, method="compose")
    
    else:
        return concatenate_videoclips(clips, method="compose")


def build_video(
    clips_dir: Path,
    audio_path: Path,
    transcript_path: Path,
    output_path: Path,
    fps: Optional[int] = None,
    mode: str = "prod",
    test_duration: float = 60.0,
    # Text formatting options
    font_size: Optional[int] = None,
    font_family: Optional[str] = None,
    font_color: str = "white",
    font_bold: bool = True,
    stroke_color: str = "black",
    stroke_width: Optional[int] = None,
    # Watermark options
    watermark_path: Optional[Path] = None,
    watermark_opacity: float = 0.8,
    watermark_scale: float = 0.15,
    watermark_margin: int = 20,
    # Transition options
    transition: str = "fadeblack",
    transition_duration: float = 0.5,
) -> None:
    # Load and concat video clips with transitions
    clips = gather_clips(clips_dir)
    base = apply_transitions(clips, transition, transition_duration)

    # Load subtitles
    segments = parse_transcript_file(transcript_path)
    if not segments:
        print("Note: No parsed subtitle segments. Proceeding without text overlays.")

    # Limit processing for test mode
    if mode.lower() == "test":
        max_dur = min(float(test_duration), float(base.duration))
        # Trim base to speed up processing
        base = base.subclip(0, max_dur)
        # Keep only subtitle segments that start before max_dur and clamp their end
        trimmed_segments: List[VideoSegment] = []
        for s, e, txt in segments:
            if s >= max_dur:
                break
            trimmed_segments.append((s, min(e, max_dur), txt))
        segments = trimmed_segments

    # Build text overlays
    subs: List[TextClip] = []
    for s, e, txt in segments:
        tc = make_subtitle_clip(
            txt, s, e, base_w=base.w, base_h=base.h,
            font_size=font_size,
            font_family=font_family,
            font_color=font_color,
            font_bold=font_bold,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        if tc is not None:
            subs.append(tc)

    # Load and fit audio
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio = AudioFileClip(str(audio_path))
    audio_fitted = fit_audio_to_duration(audio, base.duration)

    # Composite subtitles and optional watermark over base video
    layers = [base] + subs

    if watermark_path is not None:
        print(f"Adding watermark: {watermark_path}")
        # Use the same approach as the reference implementation
        wm = (
            ImageClip(str(watermark_path))
            .resize(height=80)  # Keep aspect ratio, ~80px height (YouTube style)
            .set_opacity(watermark_opacity)
            .set_duration(base.duration)
            .set_position(("right", "bottom"))  # Built-in positioning like reference
        )
        print(f"Watermark added with 80px height and {watermark_opacity} opacity")
        layers.append(wm)

    final = CompositeVideoClip(layers)
    final = final.set_audio(audio_fitted)

    # Determine fps
    out_fps = fps or (base.fps if hasattr(base, "fps") and base.fps else 30)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=out_fps,
        threads=4,
        temp_audiofile=str(output_path.with_suffix(".temp-audio.m4a")),
        remove_temp=True,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Combine clips, overlay transcript, watermark, and set external audio.")
    parser.add_argument("--clips-dir", type=str, default=str(Path("clips")), help="Directory containing video clips")
    parser.add_argument("--audio", type=str, default=str(Path("analyzed_filepathX75.wav")), help="Path to WAV audio file")
    parser.add_argument("--transcript", type=str, default=str(Path("transcript_output.txt")), help="Path to transcript file")
    parser.add_argument("--output", type=str, default=str(Path("output.mp4")), help="Output video file path")
    parser.add_argument("--fps", type=int, default=None, help="Output frames per second (default: base video fps or 30)")
    parser.add_argument("--mode", type=str, choices=["prod", "test"], default="prod", help="Mode: prod (full render) or test (limit duration)")
    parser.add_argument("--test-duration", type=float, default=60.0, help="Max duration in seconds when mode=test (default: 60)")

    # Text formatting options
    parser.add_argument("--font-size", type=int, default=None, help="Subtitle font size (default scales with video height)")
    parser.add_argument("--font-family", type=str, default=None, help="Subtitle font family (e.g., 'Arial', 'Roboto', 'Open Sans') - supports Google Fonts")
    parser.add_argument("--font-color", type=str, default="white", help="Subtitle font color (default: white)")
    parser.add_argument("--font-bold", action="store_true", help="Use bold font variants when available")
    parser.add_argument("--stroke-color", type=str, default="black", help="Outline color for subtitles (default: black)")
    parser.add_argument("--stroke-width", type=int, default=None, help="Outline thickness for subtitles (default scales with video height)")

    # Watermark options
    parser.add_argument("--watermark", type=str, default="watermark.jpeg", help="Path to watermark image (default: watermark.jpeg)")
    parser.add_argument("--watermark-opacity", type=float, default=0.35, help="Opacity of watermark (0-1, default 0.35 - YouTube style)")
    parser.add_argument("--watermark-scale", type=float, default=0.06, help="Watermark width as fraction of video width (default 0.06 - YouTube style)")
    parser.add_argument("--watermark-margin", type=int, default=15, help="Margin from bottom-right corner in pixels (default 15)")
    parser.add_argument("--no-watermark", action="store_true", help="Disable watermark even if watermark.jpeg exists")

    # Transition options
    parser.add_argument("--transition", type=str, choices=["none", "fadeblack", "crossfade"], default="fadeblack", help="Transition type between clips")
    parser.add_argument("--transition-duration", type=float, default=0.5, help="Transition duration in seconds (default 0.5)")

    # Utility options
    parser.add_argument("--list-fonts", action="store_true", help="List popular Google Fonts and exit")

    args = parser.parse_args()

    # Handle font listing
    if args.list_fonts:
        print("Fetching popular Google Fonts...")
        fonts_data = get_google_fonts_list()
        if fonts_data:
            print("\nPopular Google Fonts (top 50):")
            print("=" * 40)
            for i, font in enumerate(fonts_data.get('items', [])[:50], 1):
                family = font['family']
                category = font.get('category', 'unknown')
                print(f"{i:2d}. {family} ({category})")
            print("\nUsage: --font-family 'Font Name'")
            print("Example: --font-family 'Roboto' --font-bold")
        else:
            print("Could not fetch Google Fonts list. Check your internet connection.")
        return

    clips_dir = Path(args.clips_dir)
    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    output_path = Path(args.output)

    build_video(
        clips_dir=clips_dir,
        audio_path=audio_path,
        transcript_path=transcript_path,
        output_path=output_path,
        fps=args.fps,
        mode=args.mode,
        test_duration=args.test_duration,
        # text formatting
        font_size=args.font_size,
        font_family=args.font_family,
        font_color=args.font_color,
        font_bold=bool(args.font_bold),
        stroke_color=args.stroke_color,
        stroke_width=args.stroke_width,
        # watermark
        watermark_path=None if args.no_watermark else (Path(args.watermark) if Path(args.watermark).exists() else None),
        watermark_opacity=args.watermark_opacity,
        watermark_scale=args.watermark_scale,
        watermark_margin=args.watermark_margin,
        # transitions
        transition=args.transition,
        transition_duration=args.transition_duration,
    )


if __name__ == "__main__":
    main()
