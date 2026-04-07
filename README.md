# ASCII Video Player

> Convert video files to ASCII art animation with real-time playback

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Description

**ASCII Video Player** is a desktop application that converts video files into ASCII art animations. It processes video frames in real-time, mapping pixel intensities to characters to create a text-based representation of the video.

### Features

- **Real-time video playback** as ASCII art
- **Adjustable settings**: Width, font size, brightness, contrast, and character set
- **Performance optimizations**: Frame caching and LRU cache for converted frames
- **Playback controls**: Play, pause, stop, seek forward/backward (5-second jumps)
- **Keyboard shortcuts** for all major functions
- **Persistent settings** saved to `~/.ascii_video_player/config.json`
- **Scrollable display** for large ASCII outputs
- **Performance metrics** showing conversion time per frame

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Play/Pause |
| `Esc` | Stop |
| `Left Arrow` | Seek back 5 seconds |
| `Right Arrow` | Seek forward 5 seconds |
| `Ctrl +` | Increase font size (zoom in) |
| `Ctrl -` | Decrease font size (zoom out) |
| `Ctrl O` | Open video file |
| `Ctrl Q` | Quit application |

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ascii-video-player.git
cd ascii-video-player
```

2. **Install dependencies**
 ```powershell
   pip install opencv-python numpy
   ```
**Note: tkinter is included with most Python installations. If missing, install via your package manager:**

Ubuntu/Debian: ``sudo apt-get install python3-tk``

macOS: ``brew install python-tk``



Fedora: ``sudo dnf install python3-tkinter``

Arch Linux: ``sudo pacman -S tk``

Windows: Included with Python installere 
