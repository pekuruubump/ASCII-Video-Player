import cv2 # You need to intall this
import numpy as np # And this so it works properly.

import tkinter as tk  # Up to this point everything is pre-installed
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

# Configure logging to both file and console for debugging user issues
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
    """User-configurable settings with validation.
    
    All settings are clamped to reasonable ranges because extreme values
    can make the video unreadable or cause performance issues.
    """
    width: int = 100
    font_size: int = 10
    brightness: float = 1.0
    contrast: float = 1.0
    chars: str = " .:-=+*#%@"
    cache_size: int = 100
    target_fps: int = 30

    def __post_init__(self):
        # Width affects both quality and performance - limit to sane values
        self.width = max(40, min(300, self.width))
        # Very small fonts are unreadable, very large break layout
        self.font_size = max(6, min(24, self.font_size))
        self.brightness = max(0.0, min(2.0, self.brightness))
        self.contrast = max(0.0, min(3.0, self.contrast))
        # Cache too small = recomputation; too large = memory bloat
        self.cache_size = max(10, min(500, self.cache_size))
        # 15-60 FPS is the range most displays and videos actually use
        self.target_fps = max(15, min(60, self.target_fps))

    def to_dict(self) -> dict:
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
        # Only load fields that exist - ignore unknown JSON keys
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ThreadSafeCache:
    """LRU cache for ASCII conversions.
    
    Converting frames to ASCII is expensive. This cache stores recent results
    keyed by a hash of the frame + settings. Thread-safe because the converter
    runs in a background thread while UI updates happen in the main thread.
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order = deque(maxlen=max_size)  # Tracks LRU order
        self._lock = threading.RLock()  # RLock allows same-thread reentrancy

    def _generate_key(self, frame: np.ndarray, settings_hash: int) -> str:
        """Create a cache key from a downsampled frame hash.
        
        Downsampling to 32x24 before hashing prevents minor frame differences
        (like a single pixel changing) from generating completely different keys.
        This improves cache hit rate without noticeable quality loss.
        """
        small_frame = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
        frame_bytes = small_frame.tobytes()
        frame_hash = hashlib.md5(frame_bytes).hexdigest()[:16]
        return f"{frame_hash}_{settings_hash}"

    def get(self, frame: np.ndarray, settings_hash: int) -> Optional[str]:
        with self._lock:
            key = self._generate_key(frame, settings_hash)
            if key in self._cache:
                # Move to end of access order (most recently used)
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                return self._cache[key]
            return None

    def put(self, frame: np.ndarray, settings_hash: int, ascii_art: str):
        with self._lock:
            key = self._generate_key(frame, settings_hash)

            # Evict oldest if at capacity and this is a new key
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.popleft()
                self._cache.pop(oldest_key, None)

            self._cache[key] = ascii_art
            if key not in self._access_order:
                self._access_order.append(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_size(self) -> int:
        with self._lock:
            return len(self._cache)


class ASCIIConverter:
    """Converts video frames to ASCII art.
    
    This is the core transformation engine. It handles:
    1. Grayscale conversion
    2. Brightness/contrast adjustment
    3. Aspect ratio correction (characters are taller than they are wide)
    4. Mapping pixel intensities to characters
    """
    def __init__(self):
        self.settings = VideoSettings()
        self.cache: Optional[ThreadSafeCache] = None
        self._settings_lock = threading.RLock()
        self._init_cache()

    def _init_cache(self):
        with self._settings_lock:
            self.cache = ThreadSafeCache(max_size=self.settings.cache_size)

    def update_settings(self, new_settings: VideoSettings):
        """Reset cache when settings change because old cached frames are now invalid."""
        with self._settings_lock:
            if self.settings != new_settings:
                self.settings = new_settings
                self._init_cache()
                logger.debug("Settings updated and cache reset")

    def _get_settings_hash(self) -> int:
        """Hash only the settings that affect the conversion output."""
        with self._settings_lock:
            return hash((
                self.settings.width,
                self.settings.brightness,
                self.settings.contrast,
                self.settings.chars
            ))

    def frame_to_ascii(self, frame: np.ndarray) -> str:
        """Convert a single BGR frame to ASCII art string."""
        settings_hash = self._get_settings_hash()

        # Fast path: return cached result if available
        cached = self.cache.get(frame, settings_hash)
        if cached is not None:
            return cached

        # Convert BGR to grayscale (CV2 reads in BGR by default)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply brightness/contrast BEFORE resizing for better quality
        # This formula: output = alpha * input + beta where alpha=contrast, beta controls brightness
        if self.settings.brightness != 1.0 or self.settings.contrast != 1.0:
            alpha = self.settings.contrast
            beta = (self.settings.brightness - 1.0) * 255
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Aspect ratio correction: terminal characters are ~2.2x taller than wide
        # Without this, the output image would look vertically stretched
        aspect_ratio = 0.45
        target_width = self.settings.width
        target_height = int(gray.shape[0] * target_width / gray.shape[1] * aspect_ratio)
        target_height = max(1, target_height)

        resized = cv2.resize(gray, (target_width, target_height),
                            interpolation=cv2.INTER_AREA)

        # Map each pixel's intensity (0-255) to an index in the character set
        indices = (resized.astype(np.float32) / 255.0 * (len(self.settings.chars) - 1))
        indices = np.clip(indices, 0, len(self.settings.chars) - 1).astype(np.int32)

        # Vectorized lookup: much faster than iterating pixel by pixel
        ascii_array = np.array(list(self.settings.chars))[indices]
        ascii_str = '\n'.join([''.join(row) for row in ascii_array])

        self.cache.put(frame, settings_hash, ascii_str)

        return ascii_str


class VideoProcessor(threading.Thread):
    """Background thread that reads video frames and converts them.
    
    Runs independently of the UI so video playback doesn't stutter when
    the user interacts with controls. Uses queues for thread-safe communication
    with the main thread.
    """
    def __init__(self, video_path: str, converter: ASCIIConverter):
        super().__init__(daemon=True, name="VideoProcessor")
        self.video_path = video_path
        self.converter = converter

        # Thread synchronization primitives
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_event = threading.Event()
        self._seek_position = 0
        self._state_lock = threading.RLock()

        # Communication queues with size limits to prevent memory explosion
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.settings_update_queue: queue.Queue = queue.Queue()
        self.command_queue: queue.Queue = queue.Queue()

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_pos = 0
        self.total_frames = 0
        self.fps = 0
        self._error_state = None
        self._video_loaded = False

        # Rolling window of processing times for performance monitoring
        self.processing_time = deque(maxlen=30)
        self.is_playing = False

        self._open_video()

    def _open_video(self) -> bool:
        """Initialize video capture. Returns False if video can't be opened."""
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
                    self.fps = 30  # Fallback for videos with broken FPS metadata

                self.current_frame_pos = 0
                self._video_loaded = True
                logger.info(f"Video loaded: {self.total_frames} frames, {self.fps:.1f} FPS")
                return True

        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            self._error_state = str(e)
            return False

    def run(self):
        """Main processing loop. Runs in background thread."""
        if not self._video_loaded:
            self.frame_queue.put(("error", self._error_state or "Video not loaded"))
            return

        # Target frame timing - we sleep to match video's original FPS
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033

        while not self._stop_event.is_set():
            try:
                self._process_commands()

                # Apply any pending settings changes
                try:
                    new_settings = self.settings_update_queue.get_nowait()
                    self.converter.update_settings(new_settings)
                except queue.Empty:
                    pass

                # Handle seek operations (jump to different time position)
                if self._seek_event.is_set():
                    self._perform_seek()
                    self._seek_event.clear()

                # Pause handling: don't read frames when paused
                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                start_time = time.time()

                # Read next frame (thread-safe with lock)
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

                # Convert frame to ASCII and measure performance
                conversion_start = time.time()
                ascii_art = self.converter.frame_to_ascii(frame)
                conversion_time = (time.time() - conversion_start) * 1000

                self.processing_time.append(conversion_time)
                avg_conversion = sum(self.processing_time) / len(self.processing_time) if self.processing_time else 0

                # Package frame with metadata for UI display
                frame_info = {
                    'current': self.current_frame_pos,
                    'total': self.total_frames,
                    'video_fps': self.fps,
                    'conversion_ms': conversion_time,
                    'avg_conversion_ms': avg_conversion,
                    'is_playing': self.is_playing
                }

                # Non-blocking put - drop frame if queue is full rather than stalling
                try:
                    self.frame_queue.put((ascii_art, frame_info), timeout=0.1)
                except queue.Full:
                    pass

                # Maintain correct playback speed by sleeping
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
        """Process all queued commands without blocking."""
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
        """Jump to a specific frame position.
        
        Also clears the frame queue to prevent showing stale frames
        after the seek completes.
        """
        with self._state_lock:
            if self.cap and self.cap.isOpened():
                position = max(0, min(self._seek_position, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Clear stale queued frames
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

    def _handle_end_of_video(self):
        """Handle reaching the end of the video file.
        
        Pauses playback and resets position to beginning so next play
        starts from the start, not from the end.
        """
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
        self.is_playing = True
        self._pause_event.clear()

    def pause(self):
        self.is_playing = False
        self._pause_event.set()

    def stop(self):
        """Stop playback and reset to beginning."""
        self.pause()
        with self._state_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_pos = 0

    def update_settings(self, settings: VideoSettings):
        """Queue a settings update (non-blocking, returns immediately)."""
        try:
            self.settings_update_queue.put(settings, timeout=0.1)
        except queue.Full:
            logger.warning("Settings update queue full, skipping")

    def seek(self, frame_position: int):
        self.command_queue.put({'type': 'seek', 'position': frame_position})

    def seek_relative(self, delta: int):
        self.command_queue.put({'type': 'seek_relative', 'delta': delta})

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame without advancing playback position."""
        with self._state_lock:
            if self.cap and self.cap.isOpened():
                original_pos = self.current_frame_pos
                ret, frame = self.cap.read()
                if ret:
                    # Restore original position after reading
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
                    return frame
        return None

    def get_state(self) -> dict:
        with self._state_lock:
            return {
                'is_playing': self.is_playing,
                'current_frame': self.current_frame_pos,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'error': self._error_state
            }

    def cleanup(self):
        """Release video resources and stop thread gracefully."""
        logger.info("Cleaning up video processor...")
        self._stop_event.set()
        self._pause_event.set()

        with self._state_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        # Clear queues to release any waiting threads
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Video processor cleaned up")

    def __del__(self):
        self.cleanup()


class OptimizedDisplay:
    """Tkinter widget that renders ASCII art efficiently.
    
    Uses batching and throttling to avoid updating the UI for every frame.
    The canvas + scrollbars setup allows viewing large ASCII outputs that
    exceed the window size.
    """
    def __init__(self, parent, font_size: int = 10):
        self.parent = parent
        self.font_size = font_size
        self.current_ascii = ""
        self._update_lock = threading.Lock()
        self._update_scheduled = False  # Prevents queueing multiple updates

        # Canvas provides scrollable viewport for large ASCII art
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
            fg="#00ff00",  # Classic green-on-black terminal look
            font=("Courier", font_size),
            wrap=tk.NONE,  # Don't wrap - let scrollbars handle it
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
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_ascii(self, ascii_art: str):
        """Schedule a UI update, throttling to ~60 FPS maximum.
        
        Multiple updates within the same frame are coalesced to prevent
        the UI thread from being overwhelmed.
        """
        with self._update_lock:
            if ascii_art == self.current_ascii:
                return  # No change, skip update

            self.current_ascii = ascii_art

            if not self._update_scheduled:
                self._update_scheduled = True
                # ~16ms = 60fps - fast enough for smooth animation, slow enough to not choke
                self.parent.after(16, self._perform_update)

    def _perform_update(self):
        """Actually update the UI. Runs in main thread via after()."""
        with self._update_lock:
            self._update_scheduled = False

            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, self.current_ascii)

            # Resize Text widget to exactly fit the content
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
        self.font_size = size
        self.text_widget.configure(font=("Courier", size))
        if self.current_ascii:
            self.update_ascii(self.current_ascii)

    def clear(self):
        self.text_widget.delete(1.0, tk.END)
        self.current_ascii = ""

    def destroy(self):
        self.canvas.destroy()
        self.text_widget.destroy()


class ConfigManager:
    """Persist user settings to ~/.ascii_video_player/config.json
    
    Using the user's home directory ensures settings survive app reinstalls
    and don't require admin permissions to write.
    """
    CONFIG_DIR = Path.home() / '.ascii_video_player'
    CONFIG_FILE = CONFIG_DIR / 'config.json'

    @classmethod
    def load(cls) -> VideoSettings:
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return VideoSettings.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        return VideoSettings()  # Fallback to defaults

    @classmethod
    def save(cls, settings: VideoSettings):
        try:
            cls.CONFIG_DIR.mkdir(exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(settings.to_dict(), f, indent=2)
            logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class KeyboardShortcuts:
    """Centralizes keyboard shortcut management.
    
    Keeping shortcuts in one place makes them easier to modify and
    prevents binding conflicts.
    """
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
        for key, callback_name in self.SHORTCUTS.items():
            callback = getattr(self.app, f'_on_{callback_name}', None)
            if callback:
                self._bindings[key] = self.app.root.bind(key, callback)

    def unbind_all(self):
        for key, binding_id in self._bindings.items():
            self.app.root.unbind(key, binding_id)


class ASCIIVideoPlayer:
    """Main application class - coordinates all components.
    
    This class is intentionally large because Tkinter applications typically
    have a single controller that manages UI, events, and background threads.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Video Player")
        self.root.geometry("1200x800")

        # Ensure cleanup happens even if the app crashes
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
        
        self._settings_update_timer = None
        self._settings_update_lock = threading.Lock()

        # Color scheme - matches VS Code dark theme for familiarity
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"

        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        self.keyboard_shortcuts.bind_all()

        self._load_settings_to_ui()
        self.apply_settings_real_time()

    def setup_ui(self):
        """Build all UI elements.
        
        Organized into logical sections: file controls, playback controls,
        settings panel, display area, and status bars.
        """
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
            command=lambda: self.seek_relative(-150),  # 5 seconds at 30fps = 150 frames
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
        """Settings controls with debounced updates.
        
        Debouncing (100ms delay) prevents excessive updates while the user
        is dragging sliders or spinning spinboxes.
        """
        settings_frame = tk.Frame(parent, bg=self.bg_color)
        settings_frame.pack(side=tk.RIGHT, padx=5)

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
            command=self._schedule_settings_update
        )
        width_spinbox.pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *args: self._schedule_settings_update())

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
            command=self._schedule_settings_update
        )
        font_spinbox.pack(side=tk.LEFT, padx=2)
        self.font_var.trace_add('write', lambda *args: self._schedule_settings_update())

        tk.Label(settings_frame, text="Bright:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.brightness_var = tk.DoubleVar(value=self.current_settings.brightness)
        brightness_scale = tk.Scale(
            settings_frame,
            from_=0.0,
            to=2.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=100,
            variable=self.brightness_var,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            command=lambda x: self._schedule_settings_update()
        )
        brightness_scale.pack(side=tk.LEFT, padx=2)

        tk.Label(settings_frame, text="Contrast:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.contrast_var = tk.DoubleVar(value=self.current_settings.contrast)
        contrast_scale = tk.Scale(
            settings_frame,
            from_=0.0,
            to=3.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=100,
            variable=self.contrast_var,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            command=lambda x: self._schedule_settings_update()
        )
        contrast_scale.pack(side=tk.LEFT, padx=2)

        tk.Label(settings_frame, text="Charset:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.charset_var = tk.StringVar(value=self.current_settings.chars)
        # Predefined character sets from low to high density
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
        charset_menu.bind('<<ComboboxSelected>>', lambda e: self._schedule_settings_update())

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
            command=self._schedule_settings_update
        )
        cache_spinbox.pack(side=tk.LEFT, padx=2)
        self.cache_var.trace_add('write', lambda *args: self._schedule_settings_update())

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

    def _schedule_settings_update(self):
        """Debounce settings updates to avoid overwhelming the processor.
        
        If the user drags a slider, we wait 100ms after the last change
        before applying the new settings.
        """
        with self._settings_update_lock:
            if self._settings_update_timer:
                self.root.after_cancel(self._settings_update_timer)
            
            self._settings_update_timer = self.root.after(100, self._apply_debounced_settings)

    def _apply_debounced_settings(self):
        """Actually apply the settings after debounce period."""
        with self._settings_update_lock:
            self._settings_update_timer = None
            
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
                
                if new_settings == self.current_settings:
                    return  # No actual change
                
                self.current_settings = new_settings
                
                self.converter.update_settings(new_settings)
                
                if self.display:
                    self.display.set_font_size(new_settings.font_size)
                
                if self.video_processor:
                    self.video_processor.update_settings(new_settings)
                    
                    # If video is paused, refresh the current frame with new settings
                    state = self.video_processor.get_state()
                    if not state['is_playing']:
                        self.update_current_frame_display()
                    else:
                        self.status_bar.config(text=f"Settings: {new_settings.width}x{new_settings.font_size}")
                        if hasattr(self, '_status_reset_job'):
                            self.root.after_cancel(self._status_reset_job)
                        self._status_reset_job = self.root.after(1000, 
                            lambda: self.status_bar.config(text="Playing"))
                
            except Exception as e:
                logger.error(f"Error applying settings: {e}")
                self.status_bar.config(text=f"Error: {e}")

    def reset_settings(self):
        self.current_settings = VideoSettings()
        self._load_settings_to_ui()
        self._apply_debounced_settings()
        self.status_bar.config(text="Settings reset to defaults")

    def _load_settings_to_ui(self):
        self.width_var.set(str(self.current_settings.width))
        self.font_var.set(str(self.current_settings.font_size))
        self.brightness_var.set(str(self.current_settings.brightness))
        self.contrast_var.set(str(self.current_settings.contrast))
        self.charset_var.set(self.current_settings.chars)
        self.cache_var.set(str(self.current_settings.cache_size))

    def apply_settings_real_time(self):
        self._schedule_settings_update()

    def save_settings(self):
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
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # Clean up previous video if any
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
        if self.display_update_job:
            self.root.after_cancel(self.display_update_job)
        self.update_display()

    def update_display(self):
        """Main display update loop - runs every ~16ms in the main thread.
        
        Pulls frames from the processor's queue and updates the UI.
        Processes up to 3 frames per tick to prevent queue backlog.
        """
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

                    # Update status bar with performance metrics
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

        # Schedule next update (~60 FPS)
        self.display_update_job = self.root.after(16, self.update_display)

    def toggle_playback(self):
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
        if not self.video_processor:
            return

        self.video_processor.seek_relative(delta)
        self.status_bar.config(text=f"Seeking...")

    def update_current_frame_display(self):
        """Refresh the display with current frame (used after settings changes while paused)."""
        if not self.video_processor:
            return

        state = self.video_processor.get_state()
        if state['is_playing']:
            return  # Playing will update automatically

        try:
            frame = self.video_processor.get_current_frame()
            if frame is not None:
                ascii_art = self.converter.frame_to_ascii(frame)
                if self.display:
                    self.display.update_ascii(ascii_art)
        except Exception as e:
            logger.error(f"Error updating current frame: {e}")

    # Keyboard shortcut handlers
    def _on_play_pause(self, event=None):
        self.toggle_playback()
        return "break"  # Prevents default key handling

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
        return "break"  # Not implemented yet

    def _on_volume_down(self, event=None):
        return "break"

    def _on_zoom_in(self, event=None):
        current_font = int(self.font_var.get())
        if current_font < 24:
            self.font_var.set(str(current_font + 1))
            self._schedule_settings_update()
        return "break"

    def _on_zoom_out(self, event=None):
        current_font = int(self.font_var.get())
        if current_font > 6:
            self.font_var.set(str(current_font - 1))
            self._schedule_settings_update()
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
        """Release all resources before exit.
        
        Called both by atexit and window close event to ensure
        background threads are stopped properly.
        """
        logger.info("Cleaning up application...")
        
        if self._settings_update_timer:
            try:
                self.root.after_cancel(self._settings_update_timer)
            except:
                pass

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
        self.cleanup()
        self.root.destroy()


def main():
    """Entry point with dependency checking."""
    try:
        # Check for optional dependencies (they should be installed, but just in case)
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
