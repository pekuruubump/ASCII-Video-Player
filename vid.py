import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
from collections import deque
import queue
import threading

class ASCIIVideoGUI:
    def __init__(self, root):
        """Set up the video player window with all the controls"""
        self.root = root
        self.root.title("ASCII Video Player")
        self.root.geometry("1000x700")
        
        # Player state
        self.video_path = None
        self.cap = None
        self.playing = False
        self.current_frame = None
        self.playback_thread = None
        self.stop_thread = False
        
        # Cache for ASCII conversion
        self.ascii_cache = {}
        self.cache_maxsize = 30  # Cache up to 30 frames
        
        # Thread-safe settings
        self.settings_lock = threading.Lock()
        self.settings_changed = False
        
        # Frame queue with timestamp
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # ASCII art settings 
        self.ascii_width = 100
        self.font_size = 10
        self.chars = " .:-=+*#%@"
        self.brightness_adjust = 1.0
        self.contrast_adjust = 1.0
        
        # Precompute character mappings for faster conversion
        self.char_map = np.array([ord(c) for c in self.chars])
        
        # Dark theme colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"
        
        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        
    def setup_ui(self):
        """Build all the buttons, sliders, and display area"""
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        file_frame = tk.Frame(control_frame, bg=self.bg_color)
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
            text="Open Video", 
            command=self.open_video,
            bg=self.accent_color, 
            fg="white",
            relief=tk.FLAT, 
            padx=10,
            cursor="hand2"
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        playback_frame = tk.Frame(control_frame, bg=self.bg_color)
        playback_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_play = tk.Button(
            playback_frame, 
            text="Play", 
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
            text="Stop", 
            command=self.stop_playback,
            bg="#2d2d2d", 
            fg="white",
            relief=tk.FLAT, 
            padx=10, 
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        # Settings panel
        settings_frame = tk.Frame(control_frame, bg=self.bg_color)
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        # Width control
        tk.Label(settings_frame, text="Width:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value="100")
        width_entry = tk.Entry(
            settings_frame, 
            textvariable=self.width_var, 
            width=5,
            bg="#2d2d2d", 
            fg=self.fg_color, 
            relief=tk.FLAT,
            insertbackground=self.fg_color
        )
        width_entry.pack(side=tk.LEFT, padx=2)
        
        # Font size control
        tk.Label(settings_frame, text="Font size:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value="10")
        font_entry = tk.Entry(
            settings_frame, 
            textvariable=self.font_var, 
            width=5,
            bg="#2d2d2d", 
            fg=self.fg_color, 
            relief=tk.FLAT,
            insertbackground=self.fg_color
        )
        font_entry.pack(side=tk.LEFT, padx=2)
        
        # Brightness control
        tk.Label(settings_frame, text="Brightness:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.brightness_var = tk.StringVar(value="1.0")
        brightness_entry = tk.Entry(
            settings_frame, 
            textvariable=self.brightness_var, 
            width=5,
            bg="#2d2d2d", 
            fg=self.fg_color, 
            relief=tk.FLAT
        )
        brightness_entry.pack(side=tk.LEFT, padx=2)
        
        # Live update checkbox
        self.live_update_var = tk.BooleanVar(value=True)
        self.live_update_check = tk.Checkbutton(
            settings_frame,
            text="Live Update",
            variable=self.live_update_var,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            command=self.toggle_live_update
        )
        self.live_update_check.pack(side=tk.LEFT, padx=5)
        
        self.btn_apply = tk.Button(
            settings_frame, 
            text="Apply", 
            command=self.apply_settings,
            bg="#2d2d2d", 
            fg="white",
            relief=tk.FLAT, 
            padx=10,
            cursor="hand2"
        )
        self.btn_apply.pack(side=tk.LEFT, padx=5)
        
        # Display area with optimized scrolling
        display_frame = tk.Frame(main_frame, bg="#000000")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for better performance with large text
        self.text_widget = tk.Text(
            display_frame, 
            bg="#000000", 
            fg="#00ff00",
            font=("Courier", self.font_size),
            wrap=tk.NONE,
            relief=tk.FLAT,
            highlightthickness=0,
            insertbackground="#00ff00"
        )
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = tk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.text_widget.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.text_widget.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            bg="#007acc", 
            fg="white",
            anchor=tk.W, 
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_live_update(self):
        """Enable/disable live update of settings"""
        if self.live_update_var.get():
            self.status_bar.config(text="Live update enabled - settings apply immediately")
        else:
            self.status_bar.config(text="Live update disabled - click Apply to update")
    
    def open_video(self):
        """Let user pick a video file to play"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), 
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.btn_play.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Loaded: {file_path}")
            
            # Clean up old video
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            
            # Clear cache when loading new video
            with self.settings_lock:
                self.ascii_cache.clear()
                self.settings_changed = False
            
            # Get video info for status
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)} | FPS: {fps:.1f} | Frames: {total_frames}")
    
    def get_current_settings(self):
        """Thread-safe method to get current settings"""
        with self.settings_lock:
            return {
                'width': self.ascii_width,
                'brightness': self.brightness_adjust,
                'contrast': self.contrast_adjust,
                'font_size': self.font_size
            }
    
    def frame_to_ascii_fast(self, frame):
        """Optimized frame to ASCII conversion using numpy vectorization"""
        # Get current settings thread-safely
        settings = self.get_current_settings()
        
        # Generate cache key including settings
        frame_hash = hash((frame.tobytes(), settings['width'], settings['brightness'], settings['contrast'])) % 1000000
        
        # Check cache
        if frame_hash in self.ascii_cache:
            return self.ascii_cache[frame_hash]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply brightness and contrast adjustments
        if settings['brightness'] != 1.0 or settings['contrast'] != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=settings['contrast'], beta=settings['brightness'] * 255)
        
        # Calculate dimensions with aspect ratio
        aspect_ratio = 0.55
        height = int(gray.shape[0] * settings['width'] / gray.shape[1] * aspect_ratio)
        
        # Resize image (using INTER_AREA for better quality when shrinking)
        resized = cv2.resize(gray, (settings['width'], height), interpolation=cv2.INTER_AREA)
        
        # Vectorized ASCII conversion
        indices = (resized.astype(np.float32) / 255.0 * (len(self.chars) - 1)).astype(np.int32)
        indices = np.clip(indices, 0, len(self.chars) - 1)
        
        # Convert to string efficiently
        ascii_array = np.array(list(self.chars))[indices]
        ascii_str = '\n'.join([''.join(row) for row in ascii_array])
        
        # Cache the result
        if len(self.ascii_cache) >= self.cache_maxsize:
            # Remove oldest item
            self.ascii_cache.pop(next(iter(self.ascii_cache)))
        self.ascii_cache[frame_hash] = ascii_str
        
        return ascii_str
    
    def update_display_async(self):
        """Async display update using queue"""
        try:
            while True:
                ascii_art, frame_info = self.frame_queue.get_nowait()
                
                # Update text widget efficiently
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, ascii_art)
                
                # Auto-scroll to top for better viewing
                self.text_widget.see(1.0)
                
                # Update status with FPS info
                current_time = time.time()
                self.frame_times.append(current_time - self.last_frame_time)
                self.last_frame_time = current_time
                
                if len(self.frame_times) > 0:
                    avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    self.status_bar.config(text=f"{frame_info} | Display FPS: {avg_fps:.1f}")
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        if self.playing:
            self.root.after(16, self.update_display_async)  # ~60 FPS max
    
    def video_processing_thread(self):
        """Background thread for video processing"""
        while self.playing and not self.stop_thread:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                # Loop video if needed
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                if ret:
                    # Check if settings have changed
                    with self.settings_lock:
                        if self.settings_changed:
                            self.ascii_cache.clear()
                            self.settings_changed = False
                    
                    # Convert frame to ASCII
                    start_time = time.time()
                    ascii_art = self.frame_to_ascii_fast(frame)
                    conversion_time = (time.time() - start_time) * 1000
                    
                    # Get frame info
                    frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    frame_info = f"Frame: {frame_count}/{total_frames} | Video FPS: {fps:.1f} | Conv: {conversion_time:.1f}ms"
                    
                    # Add to queue for display
                    try:
                        self.frame_queue.put_nowait((ascii_art, frame_info))
                    except queue.Full:
                        pass  # Skip frame if queue is full
                    
                    # Calculate delay for smooth playback
                    if fps > 0:
                        delay = 1.0 / fps
                        time.sleep(max(0, delay - conversion_time / 1000))
    
    def toggle_playback(self):
        """Start or pause the video"""
        if not self.playing:
            # Start playing
            self.playing = True
            self.stop_thread = False
            self.btn_play.config(text="⏸ Pause", bg="#2d2d2d")
            self.btn_stop.config(state=tk.NORMAL)
            
            # Clear queue and cache
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            with self.settings_lock:
                self.ascii_cache.clear()
                self.settings_changed = False
            
            # Start processing thread
            self.playback_thread = threading.Thread(target=self.video_processing_thread, daemon=True)
            self.playback_thread.start()
            
            # Start display update
            self.update_display_async()
        else:
            # Pause
            self.playing = False
            self.stop_thread = True
            self.btn_play.config(text="▶ Play", bg="#2d2d2d")
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)
    
    def stop_playback(self):
        """Stop playing and reset to the beginning"""
        self.playing = False
        self.stop_thread = True
        self.btn_play.config(text="▶ Play", state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        # Wait for thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        
        # Reset video
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Clear display and queue
        self.text_widget.delete(1.0, tk.END)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.status_bar.config(text="Stopped")
    
    def apply_settings(self):
        """Apply user's settings for ASCII width and font size"""
        try:
            new_width = int(self.width_var.get())
            new_font_size = int(self.font_var.get())
            new_brightness = float(self.brightness_var.get())
            
            # Clamp values
            new_width = max(40, min(300, new_width))
            new_font_size = max(6, min(24, new_font_size))
            new_brightness = max(0.0, min(2.0, new_brightness))
            
            # Update settings with thread safety
            changed = False
            with self.settings_lock:
                if (new_width != self.ascii_width or 
                    new_brightness != self.brightness_adjust or
                    new_font_size != self.font_size):
                    
                    self.ascii_width = new_width
                    self.brightness_adjust = new_brightness
                    self.font_size = new_font_size
                    self.settings_changed = True
                    changed = True
            
            # Update font in main thread
            if changed:
                self.text_widget.configure(font=("Courier", new_font_size))
                self.status_bar.config(text=f"Settings applied: Width={new_width}, Font={new_font_size}, Brightness={new_brightness:.1f}")
                
                # Update display immediately if not playing
                if not self.playing and self.cap and self.cap.isOpened():
                    self.root.after(100, self.update_current_frame_display)
            else:
                self.status_bar.config(text="Settings unchanged")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid settings values! Please use numbers only.")
    
    def update_current_frame_display(self):
        """Update display with current settings when video is paused"""
        if not self.playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Go back one frame to maintain position
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                
                ascii_art = self.frame_to_ascii_fast(frame)
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, ascii_art)
                self.text_widget.see(1.0)
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_thread = True
        if self.cap:
            self.cap.release()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ASCIIVideoGUI(root)
    root.mainloop()
