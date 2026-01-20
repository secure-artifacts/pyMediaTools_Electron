# pyMediaTools Electron - Dependencies

This app uses a Python backend and FFmpeg tools. The app can try to install
Python packages automatically on first launch, but you can also install
everything manually using the links and commands below.

## Required downloads

1) Python 3
- https://www.python.org/downloads/

2) FFmpeg (includes ffprobe)
- https://ffmpeg.org/download.html
- Windows builds: https://www.gyan.dev/ffmpeg/builds/
- macOS builds: https://evermeet.cx/ffmpeg/

## Optional package managers

macOS (Homebrew)
- https://brew.sh/
- FFmpeg formula: https://formulae.brew.sh/formula/ffmpeg

Windows (Chocolatey)
- https://chocolatey.org/
- FFmpeg: https://community.chocolatey.org/packages/ffmpeg

Windows (Scoop)
- https://scoop.sh/
- FFmpeg: https://scoop.sh/#/apps?q=ffmpeg

## Python packages

The backend uses these Python packages:
- flask
- flask-cors
- requests
- pydub
- diff-match-patch
- pysrt
- yt-dlp

Manual install (all at once):

```bash
python3 -m pip install --user --upgrade flask flask-cors requests pydub diff-match-patch pysrt yt-dlp
```

If you are on Windows, replace `python3` with `python`.

