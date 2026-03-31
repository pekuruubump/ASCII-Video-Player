import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import os
from collections import deque
import queue
import atexit
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import json
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ascii_player.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class VideoSettings:
    """Immutable settings class with validation and serialization"""
    width: int = 100
    font_size: int = 10
    brightness: float = 1.0
    contrast: float = 1.0
    chars: str = " .:-=+*#%@"
    cache_size: int = 100
    target_fps: int = 30

    def __post_init__(self):
        """Validate and normalize settings"""
        self.width = max(40, min(300, self.width))
        self.font_size = max(6, min(24, self.font_size))
        self.brightness = max(0.0, min(2.0, self.brightness))
        self.contrast = max(0.0, min(3.0, self.contrast))
        self.cache_size = max(10, min(500, self.cache_size))
        self.target_fps = max(15, min(60, self.target_fps))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'width': self.width,
            'font_size': self.font_size,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'chars': self.chars,
            'cache_size': self.cache_size,
            'target_fps': self.target_fps
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoSettings':
        """Create settings from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ThreadSafeCache:
    """Thread-safe LRU cache for ASCII conversions"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def _generate_key(self, frame: np.ndarray, settings_hash: int) -> str:
        """Generate cache key from frame thumbnail and settings"""
        small_frame = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
        frame_bytes = small_frame.tobytes()
        frame_hash = hashlib.md5(frame_bytes).hexdigest()[:16]
        return f"{frame_hash}_{settings_hash}"

    def get(self, frame: np.ndarray, settings_hash: int) -> Optional[str]:
        """Get cached ASCII art if available"""
        with self._lock:
            key = self._generate_key(frame, settings_hash)
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                return self._cache[key]
            return None

    def put(self, frame: np.ndarray, settings_hash: int, ascii_art: str):
        """Cache ASCII art with LRU eviction"""
        with self._lock:
            key = self._generate_key(frame, settings_hash)

            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.popleft()
                self._cache.pop(oldest_key, None)

            self._cache[key] = ascii_art
            if key not in self._access_order:
                self._access_order.append(key)

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)


class ASCIIConverter:
    """Optimized ASCII converter with thread-safe caching"""

    def __init__(self):
        self.settings = VideoSettings()
        self.cache: Optional[ThreadSafeCache] = None
        self._settings_lock = threading.RLock()
        self._init_cache()

    def _init_cache(self):
        """Initialize cache with current settings"""
        with self._settings_lock:
            self.cache = ThreadSafeCache(max_size=self.settings.cache_size)

    def update_settings(self, new_settings: VideoSettings):
        """Update converter settings and reset cache"""
        with self._settings_lock:
            if self.settings != new_settings:
                self.settings = new_settings
                self._init_cache()
                logger.debug("Settings updated and cache reset")

    def _get_settings_hash(self) -> int:
        """Generate hash of current settings"""
        with self._settings_lock:
            return hash((
                self.settings.width,
                self.settings.brightness,
                self.settings.contrast,
                self.settings.chars
            ))

    def frame_to_ascii(self, frame: np.ndarray) -> str:
        """Convert frame to ASCII art with caching"""
        settings_hash = self._get_settings_hash()

        cached = self.cache.get(frame, settings_hash)
        if cached is not None:
            return cached

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self.settings.brightness != 1.0 or self.settings.contrast != 1.0:
            alpha = self.settings.contrast
            beta = (self.settings.brightness - 1.0) * 255
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Adjusted aspect ratio for better proportions
        aspect_ratio = 0.45
        target_width = self.settings.width
        target_height = int(gray.shape[0] * target_width / gray.shape[1] * aspect_ratio)
        target_height = max(1, target_height)

        resized = cv2.resize(gray, (target_width, target_height),
                            interpolation=cv2.INTER_AREA)

        indices = (resized.astype(np.float32) / 255.0 * (len(self.settings.chars) - 1))
        indices = np.clip(indices, 0, len(self.settings.chars) - 1).astype(np.int32)

        ascii_array = np.array(list(self.settings.chars))[indices]
        ascii_str = '\n'.join([''.join(row) for row in ascii_array])

        self.cache.put(frame, settings_hash, ascii_str)

        return ascii_str


class VideoProcessor(threading.Thread):
    """Dedicated thread for video processing with proper synchronization"""

    def __init__(self, video_path: str, converter: ASCIIConverter):
        super().__init__(daemon=True, name="VideoProcessor")
        self.video_path = video_path
        self.converter = converter

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_event = threading.Event()
        self._seek_position = 0
        self._state_lock = threading.RLock()

        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.settings_update_queue: queue.Queue = queue.Queue()
        self.command_queue: queue.Queue = queue.Queue()

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_pos = 0
        self.total_frames = 0
        self.fps = 0
        self._error_state = None
        self._video_loaded = False

        self.processing_time = deque(maxlen=30)
        self.is_playing = False

        self._open_video()

    def _open_video(self) -> bool:
        """Open video capture and populate metadata."""
        try:
            with self._state_lock:
                if self.cap is not None:
                    self.cap.release()

                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self._error_state = f"Failed to open video: {self.video_path}"
                    return False

                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 30

                self.current_frame_pos = 0
                self._video_loaded = True
                logger.info(f"Video loaded: {self.total_frames} frames, {self.fps:.1f} FPS")
                return True

        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            self._error_state = str(e)
            return False

    def run(self):
        """Main processing loop with proper synchronization"""
        if not self._video_loaded:
            self.frame_queue.put(("error", self._error_state or "Video not loaded"))
            return

        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033

        while not self._stop_event.is_set():
            try:
                self._process_commands()

                try:
                    new_settings = self.settings_update_queue.get_nowait()
                    self.converter.update_settings(new_settings)
                except queue.Empty:
                    pass

                if self._seek_event.is_set():
                    self._perform_seek()
                    self._seek_event.clear()

                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                start_time = time.time()

                with self._state_lock:
                    if not self.cap or not self.cap.isOpened():
                        time.sleep(0.1)
                        continue
                    ret, frame = self.cap.read()

                if ret:
                    with self._state_lock:
                        self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if not ret:
                    self._handle_end_of_video()
                    continue

                conversion_start = time.time()
                ascii_art = self.converter.frame_to_ascii(frame)
                conversion_time = (time.time() - conversion_start) * 1000

                self.processing_time.append(conversion_time)
                avg_conversion = sum(self.processing_time) / len(self.processing_time) if self.processing_time else 0

                frame_info = {
                    'current': self.current_frame_pos,
                    'total': self.total_frames,
                    'video_fps': self.fps,
                    'conversion_ms': conversion_time,
                    'avg_conversion_ms': avg_conversion,
                    'is_playing': self.is_playing
                }

                try:
                    self.frame_queue.put((ascii_art, frame_info), timeout=0.1)
                except queue.Full:
                    pass

                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self._error_state = str(e)
                self.frame_queue.put(("error", f"Processing error: {e}"))
                time.sleep(0.5)

    def _process_commands(self):
        """Process queued commands"""
        try:
            while True:
                command = self.command_queue.get_nowait()
                cmd_type = command.get('type')

                if cmd_type == 'seek':
                    self._seek_position = command.get('position', 0)
                    self._seek_event.set()
                elif cmd_type == 'seek_relative':
                    with self._state_lock:
                        new_pos = self.current_frame_pos + command.get('delta', 0)
                        self._seek_position = max(0, min(new_pos, self.total_frames - 1))
                        self._seek_event.set()
                elif cmd_type == 'set_playing':
                    self.is_playing = command.get('state', False)

        except queue.Empty:
            pass

    def _perform_seek(self):
        """Perform seek operation safely"""
        with self._state_lock:
            if self.cap and self.cap.isOpened():
                position = max(0, min(self._seek_position, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

    def _handle_end_of_video(self):
        """Handle reaching end of video"""
        with self._state_lock:
            if self.current_frame_pos >= self.total_frames - 1:
                self.is_playing = False
                self._pause_event.set()
                self.current_frame_pos = 0
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_queue.put(("end", "Video ended"))
                logger.info("Video playback completed")

    def play(self):
        """Start or resume playback"""
        with self._state_lock:
            self.is_playing = True
        self._pause_event.clear()

    def pause(self):
        """Pause playback"""
        with self._state_lock:
            self.is_playing = False
        self._pause_event.set()

    def stop(self):
        """Stop playback and reset"""
        self.pause()
        with self._state_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_pos = 0

    def update_settings(self, settings: VideoSettings):
        """Queue settings update"""
        try:
            self.settings_update_queue.put(settings, timeout=0.1)
        except queue.Full:
            logger.warning("Settings update queue full, skipping")

    def seek(self, frame_position: int):
        """Seek to specific frame"""
        self.command_queue.put({'type': 'seek', 'position': frame_position})

    def seek_relative(self, delta: int):
        """Seek relative to current position"""
        self.command_queue.put({'type': 'seek_relative', 'delta': delta})

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame without modifying position"""
        with self._state_lock:
            if self.cap and self.cap.isOpened():
                original_pos = self.current_frame_pos
                ret, frame = self.cap.read()
                if ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
                    return frame
        return None

    def get_state(self) -> dict:
        """Get current state safely"""
        with self._state_lock:
            return {
                'is_playing': self.is_playing,
                'current_frame': self.current_frame_pos,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'error': self._error_state
            }

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up video processor...")
        self._stop_event.set()
        self._pause_event.set()

        with self._state_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Video processor cleaned up")

    def __del__(self):
        self.cleanup()


class OptimizedDisplay:
    """Manages ASCII art display with optimized rendering"""

    def __init__(self, parent, font_size: int = 10):
        self.parent = parent
        self.font_size = font_size
        self.current_ascii = ""
        self._update_lock = threading.Lock()
        self._update_scheduled = False

        self.canvas = tk.Canvas(
            parent,
            bg="#000000",
            highlightthickness=0,
            cursor="arrow"
        )

        self.v_scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.text_frame = tk.Frame(self.canvas, bg="#000000")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.text_frame, anchor="nw")

        self.text_widget = tk.Text(
            self.text_frame,
            bg="#000000",
            fg="#00ff00",
            font=("Courier", font_size),
            wrap=tk.NONE,
            relief=tk.FLAT,
            highlightthickness=0,
            insertbackground="#00ff00",
            height=1,
            width=1,
            state=tk.NORMAL
        )
        self.text_widget.pack()

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.text_frame.bind('<Configure>', self._on_frame_configure)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_canvas_configure(self, event):
        """Update scroll region when canvas resizes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_frame_configure(self, event):
        """Update scroll region when frame content changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_ascii(self, ascii_art: str):
        """Update displayed ASCII art with throttling"""
        with self._update_lock:
            if ascii_art == self.current_ascii:
                return

            self.current_ascii = ascii_art

            if not self._update_scheduled:
                self._update_scheduled = True
                self.parent.after(16, self._perform_update)

    def _perform_update(self):
        """Perform actual text widget update"""
        with self._update_lock:
            self._update_scheduled = False

            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, self.current_ascii)

            lines = self.current_ascii.split('\n')
            max_line_length = max(len(line) for line in lines) if lines else 0
            line_count = len(lines)

            self.text_widget.configure(
                width=max_line_length,
                height=line_count
            )

            self.text_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def set_font_size(self, size: int):
        """Change font size"""
        self.font_size = size
        self.text_widget.configure(font=("Courier", size))
        if self.current_ascii:
            self.update_ascii(self.current_ascii)

    def clear(self):
        """Clear display"""
        self.text_widget.delete(1.0, tk.END)
        self.current_ascii = ""

    def destroy(self):
        """Clean up"""
        self.canvas.destroy()
        self.text_widget.destroy()


class ConfigManager:
    """Manages application configuration"""

    CONFIG_DIR = Path.home() / '.ascii_video_player'
    CONFIG_FILE = CONFIG_DIR / 'config.json'

    @classmethod
    def load(cls) -> VideoSettings:
        """Load settings from config file"""
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return VideoSettings.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        return VideoSettings()

    @classmethod
    def save(cls, settings: VideoSettings):
        """Save settings to config file"""
        try:
            cls.CONFIG_DIR.mkdir(exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(settings.to_dict(), f, indent=2)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class KeyboardShortcuts:
    """Manages keyboard shortcuts"""

    SHORTCUTS = {
        '<space>': 'play_pause',
        '<Escape>': 'stop',
        '<Left>': 'seek_back',
        '<Right>': 'seek_forward',
        '<Up>': 'volume_up',
        '<Down>': 'volume_down',
        '<Control-plus>': 'zoom_in',
        '<Control-minus>': 'zoom_out',
        '<Control-o>': 'open_file',
        '<Control-q>': 'quit',
        '<Key>': 'handle_key'
    }

    def __init__(self, app):
        self.app = app
        self._bindings = {}

    def bind_all(self):
        """Bind all keyboard shortcuts"""
        for key, callback_name in self.SHORTCUTS.items():
            callback = getattr(self.app, f'_on_{callback_name}', None)
            if callback:
                self._bindings[key] = self.app.root.bind(key, callback)

    def unbind_all(self):
        """Unbind all keyboard shortcuts"""
        for key, binding_id in self._bindings.items():
            self.app.root.unbind(key, binding_id)


class ASCIIVideoPlayer:
    """Main application class with proper resource management"""

    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Video Player")
        self.root.geometry("1200x800")

        atexit.register(self.cleanup)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.converter = ASCIIConverter()
        self.video_processor: Optional[VideoProcessor] = None
        self.display: Optional[OptimizedDisplay] = None
        self.config = ConfigManager.load()
        self.current_settings = self.config
        self.keyboard_shortcuts = KeyboardShortcuts(self)

        self.video_path: Optional[str] = None
        self.display_update_job: Optional[str] = None

        self.ui_lock = threading.Lock()

        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"

        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        self.keyboard_shortcuts.bind_all()

        self._load_settings_to_ui()
        self.apply_settings_real_time()

    def setup_ui(self):
        """Build UI components"""
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.setup_file_controls(control_frame)
        self.setup_playback_controls(control_frame)
        self.setup_settings_panel(control_frame)

        display_frame = tk.Frame(main_frame, bg="#000000")
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.display = OptimizedDisplay(display_frame, font_size=self.current_settings.font_size)

        self.status_bar = tk.Label(
            self.root,
            text="Ready - Press Ctrl+O to open video",
            bg=self.accent_color,
            fg="white",
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.info_label = tk.Label(
            self.root,
            text="",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 9),
            anchor=tk.W,
            padx=10
        )
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_file_controls(self, parent):
        """Setup file selection UI"""
        file_frame = tk.Frame(parent, bg=self.bg_color)
        file_frame.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 10)
        )
        self.file_label.pack(side=tk.LEFT, padx=5)

        self.btn_open = tk.Button(
            file_frame,
            text="Open Video (Ctrl+O)",
            command=self.open_video,
            bg=self.accent_color,
            fg="white",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2"
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)

    def setup_playback_controls(self, parent):
        """Setup playback control buttons"""
        playback_frame = tk.Frame(parent, bg=self.bg_color)
        playback_frame.pack(side=tk.LEFT, padx=20)

        self.btn_play = tk.Button(
            playback_frame,
            text="▶ Play (Space)",
            command=self.toggle_playback,
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.btn_play.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(
            playback_frame,
            text="⏹ Stop (Esc)",
            command=self.stop_playback,
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_seek_back = tk.Button(
            playback_frame,
            text="◀◀ 5s",
            command=lambda: self.seek_relative(-150),
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=5,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.btn_seek_back.pack(side=tk.LEFT, padx=2)

        self.btn_seek_forward = tk.Button(
            playback_frame,
            text="5s ▶▶",
            command=lambda: self.seek_relative(150),
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=5,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.btn_seek_forward.pack(side=tk.LEFT, padx=2)

        progress_frame = tk.Frame(playback_frame, bg=self.bg_color)
        progress_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(progress_frame, text="Progress:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        self.position_label = tk.Label(progress_frame, text="0/0", bg=self.bg_color, fg=self.fg_color, width=12)
        self.position_label.pack(side=tk.LEFT, padx=5)

    def setup_settings_panel(self, parent):
        """Setup settings controls with real-time updates"""
        settings_frame = tk.Frame(parent, bg=self.bg_color)
        settings_frame.pack(side=tk.RIGHT, padx=5)

        # Width control
        tk.Label(settings_frame, text="Width:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value=str(self.current_settings.width))
        width_spinbox = tk.Spinbox(
            settings_frame,
            from_=40,
            to=300,
            textvariable=self.width_var,
            width=5,
            bg="#2d2d2d",
            fg=self.fg_color,
            relief=tk.FLAT,
            command=self.apply_settings_real_time
        )
        width_spinbox.pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *args: self.apply_settings_real_time())

        # Font size control
        tk.Label(settings_frame, text="Font:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value=str(self.current_settings.font_size))
        font_spinbox = tk.Spinbox(
            settings_frame,
            from_=6,
            to=24,
            textvariable=self.font_var,
            width=5,
            bg="#2d2d2d",
            fg=self.fg_color,
            relief=tk.FLAT,
            command=self.apply_settings_real_time
        )
        font_spinbox.pack(side=tk.LEFT, padx=2)
        self.font_var.trace_add('write', lambda *args: self.apply_settings_real_time())

        # Brightness control
        tk.Label(settings_frame, text="Bright:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.brightness_var = tk.StringVar(value=str(self.current_settings.brightness))
        brightness_scale = tk.Scale(
            settings_frame,
            from_=0.0,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=100,
            variable=self.brightness_var,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            command=lambda x: self.apply_settings_real_time()
        )
        brightness_scale.pack(side=tk.LEFT, padx=2)

        # Contrast control
        tk.Label(settings_frame, text="Contrast:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.contrast_var = tk.StringVar(value=str(self.current_settings.contrast))
        contrast_scale = tk.Scale(
            settings_frame,
            from_=0.0,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=100,
            variable=self.contrast_var,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            command=lambda x: self.apply_settings_real_time()
        )
        contrast_scale.pack(side=tk.LEFT, padx=2)

        # Character set selection
        tk.Label(settings_frame, text="Charset:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.charset_var = tk.StringVar(value=self.current_settings.chars)
        charset_options = [
            " .:-=+*#%@",
            " .-:=+*#%@",
            " .,:;i1tfLCG08@",
            " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        ]
        charset_menu = ttk.Combobox(
            settings_frame,
            textvariable=self.charset_var,
            values=charset_options,
            width=15,
            state="readonly"
        )
        charset_menu.pack(side=tk.LEFT, padx=2)
        charset_menu.bind('<<ComboboxSelected>>', lambda e: self.apply_settings_real_time())

        # Cache size control
        tk.Label(settings_frame, text="Cache:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.cache_var = tk.StringVar(value=str(self.current_settings.cache_size))
        cache_spinbox = tk.Spinbox(
            settings_frame,
            from_=10,
            to=500,
            textvariable=self.cache_var,
            width=5,
            bg="#2d2d2d",
            fg=self.fg_color,
            relief=tk.FLAT,
            command=self.apply_settings_real_time
        )
        cache_spinbox.pack(side=tk.LEFT, padx=2)
        self.cache_var.trace_add('write', lambda *args: self.apply_settings_real_time())

        # Save button
        self.btn_save = tk.Button(
            settings_frame,
            text="Save Settings",
            command=self.save_settings,
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2"
        )
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Reset button
        self.btn_reset = tk.Button(
            settings_frame,
            text="Reset",
            command=self.reset_settings,
            bg="#2d2d2d",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2"
        )
        self.btn_reset.pack(side=tk.LEFT, padx=2)

    def reset_settings(self):
        """Reset settings to defaults"""
        self.current_settings = VideoSettings()
        self._load_settings_to_ui()
        self.apply_settings_real_time()
        self.status_bar.config(text="Settings reset to defaults")

    def _load_settings_to_ui(self):
        """Load saved settings into UI controls"""
        self.width_var.set(str(self.current_settings.width))
        self.font_var.set(str(self.current_settings.font_size))
        self.brightness_var.set(str(self.current_settings.brightness))
        self.contrast_var.set(str(self.current_settings.contrast))
        self.charset_var.set(self.current_settings.chars)
        self.cache_var.set(str(self.current_settings.cache_size))

    def apply_settings_real_time(self):
        """Apply settings in real-time as user changes them"""
        try:
            new_settings = VideoSettings(
                width=int(self.width_var.get()),
                font_size=int(self.font_var.get()),
                brightness=float(self.brightness_var.get()),
                contrast=float(self.contrast_var.get()),
                chars=self.charset_var.get(),
                cache_size=int(self.cache_var.get()),
                target_fps=self.current_settings.target_fps
            )

            self.current_settings = new_settings
            self.converter.update_settings(new_settings)

            if self.display:
                self.display.set_font_size(new_settings.font_size)

            if self.video_processor:
                self.video_processor.update_settings(new_settings)

                state = self.video_processor.get_state()
                if not state['is_playing']:
                    self.update_current_frame_display()

            self.status_bar.config(text=f"Settings applied")
            if hasattr(self, '_status_reset_job'):
                self.root.after_cancel(self._status_reset_job)
            self._status_reset_job = self.root.after(2000, lambda: self.status_bar.config(text="Ready"))

        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            self.status_bar.config(text=f"Error: {e}")

    def save_settings(self):
        """Save current settings to config file"""
        try:
            ConfigManager.save(self.current_settings)
            self.status_bar.config(text="Settings saved successfully")
            if hasattr(self, '_status_reset_job'):
                self.root.after_cancel(self._status_reset_job)
            self._status_reset_job = self.root.after(2000, lambda: self.status_bar.config(text="Ready"))
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self.status_bar.config(text=f"Error saving: {e}")

    def open_video(self):
        """Open video file with error handling"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.stop_playback()
        if self.video_processor:
            self.video_processor.cleanup()
            if self.video_processor.is_alive():
                self.video_processor.join(timeout=2.0)
            self.video_processor = None

        self.video_processor = VideoProcessor(file_path, self.converter)

        if not self.video_processor._video_loaded:
            messagebox.showerror("Error", f"Failed to load video:\n{self.video_processor._error_state}")
            self.video_processor = None
            return

        self.video_path = file_path
        self.file_label.config(text=os.path.basename(file_path))

        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_seek_back.config(state=tk.NORMAL)
        self.btn_seek_forward.config(state=tk.NORMAL)

        state = self.video_processor.get_state()
        self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)} | FPS: {state['fps']:.1f} | Frames: {state['total_frames']}")

        self.video_processor.start()
        self.start_display_update()
        self.update_current_frame_display()

    def start_display_update(self):
        """Start the display update loop"""
        if self.display_update_job:
            self.root.after_cancel(self.display_update_job)
        self.update_display()

    def update_display(self):
        """Update display from video processor queue"""
        if not self.video_processor:
            self.display_update_job = self.root.after(100, self.update_display)
            return

        try:
            frames_processed = 0
            while frames_processed < 3:
                try:
                    data = self.video_processor.frame_queue.get_nowait()
                except queue.Empty:
                    break

                if data[0] == "error":
                    self.status_bar.config(text=f"Error: {data[1]}")
                    break
                elif data[0] == "end":
                    self.status_bar.config(text="Video ended - Press Play to restart")
                    self.btn_play.config(text="▶ Play")
                    break
                else:
                    ascii_art, frame_info = data

                    if self.display:
                        self.display.update_ascii(ascii_art)

                    if frame_info['is_playing']:
                        status_text = f"Playing - Frame: {frame_info['current']}/{frame_info['total']} | "
                        status_text += f"FPS: {frame_info['video_fps']:.1f} | "
                        status_text += f"Conv: {frame_info['conversion_ms']:.1f}ms"
                        self.status_bar.config(text=status_text)

                    self.position_label.config(text=f"{frame_info['current']}/{frame_info['total']}")

                    progress = (frame_info['current'] / frame_info['total']) * 100 if frame_info['total'] > 0 else 0
                    self.progress_var.set(progress)

                    frames_processed += 1

        except Exception as e:
            logger.error(f"Display update error: {e}")

        self.display_update_job = self.root.after(16, self.update_display)

    def toggle_playback(self):
        """Toggle play/pause"""
        if not self.video_processor:
            return

        state = self.video_processor.get_state()

        if state['is_playing']:
            self.video_processor.pause()
            self.btn_play.config(text="▶ Play")
            self.status_bar.config(text="Paused")
        else:
            self.video_processor.play()
            self.btn_play.config(text="⏸ Pause")
            self.status_bar.config(text="Playing")

    def stop_playback(self):
        """Stop playback and reset"""
        if not self.video_processor:
            return

        self.video_processor.stop()
        self.btn_play.config(text="▶ Play")

        if self.display:
            self.display.clear()

        self.update_current_frame_display()

        self.progress_var.set(0)
        self.position_label.config(text="0/0")
        self.status_bar.config(text="Stopped")

    def seek_relative(self, delta: int):
        """Seek relative to current position"""
        if not self.video_processor:
            return

        self.video_processor.seek_relative(delta)
        self.status_bar.config(text=f"Seeking...")

    def update_current_frame_display(self):
        """Update display with current frame (for paused state)"""
        if not self.video_processor:
            return

        state = self.video_processor.get_state()
        if state['is_playing']:
            return

        try:
            frame = self.video_processor.get_current_frame()
            if frame is not None:
                ascii_art = self.converter.frame_to_ascii(frame)
                if self.display:
                    self.display.update_ascii(ascii_art)
        except Exception as e:
            logger.error(f"Error updating current frame: {e}")

    def _on_play_pause(self, event=None):
        self.toggle_playback()
        return "break"

    def _on_stop(self, event=None):
        self.stop_playback()
        return "break"

    def _on_seek_back(self, event=None):
        self.seek_relative(-150)
        return "break"

    def _on_seek_forward(self, event=None):
        self.seek_relative(150)
        return "break"

    def _on_volume_up(self, event=None):
        return "break"

    def _on_volume_down(self, event=None):
        return "break"

    def _on_zoom_in(self, event=None):
        current_font = int(self.font_var.get())
        if current_font < 24:
            self.font_var.set(str(current_font + 1))
            self.apply_settings_real_time()
        return "break"

    def _on_zoom_out(self, event=None):
        current_font = int(self.font_var.get())
        if current_font > 6:
            self.font_var.set(str(current_font - 1))
            self.apply_settings_real_time()
        return "break"

    def _on_open_file(self, event=None):
        self.open_video()
        return "break"

    def _on_quit(self, event=None):
        self.on_closing()
        return "break"

    def _on_handle_key(self, event=None):
        return "break"

    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up application...")

        if self.display_update_job:
            try:
                self.root.after_cancel(self.display_update_job)
            except:
                pass

        if self.video_processor:
            self.video_processor.cleanup()
            if self.video_processor.is_alive():
                self.video_processor.join(timeout=2.0)
            self.video_processor = None

        if self.display:
            self.display.destroy()
            self.display = None

        logger.info("Cleanup complete")

    def on_closing(self):
        """Handle window close event"""
        self.cleanup()
        self.root.destroy()


def main():
    """Entry point with error handling"""
    try:
        required_modules = ['cv2', 'numpy']
        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            messagebox.showerror(
                "Missing Dependencies",
                f"Required modules not installed:\n{', '.join(missing_modules)}\n\n"
                f"Please install them using:\npip install {' '.join(missing_modules)}"
            )
            return

        root = tk.Tk()
        app = ASCIIVideoPlayer(root)
        root.mainloop()

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")


if __name__ == "__main__":
    main()
