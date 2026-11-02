import time
import torch

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.end_time = None

    def stop(self):
        """Stop the timer and return the elapsed time in seconds."""
        if self.start_time is None:
            raise ValueError("Timer must be started before it can be stopped.")
        torch.cuda.synchronize()
        self.end_time = time.time()
        return self.end_time - self.start_time

    def elapsed(self):
        """Return the elapsed time in seconds since the timer was started,
        or the total time between start and stop if the timer has been stopped."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        if self.end_time is None:  # Timer is running
            torch.cuda.synchronize()
            now = time.time()
            return now - self.start_time
        return self.end_time - self.start_time  # Timer has been stopped

    def __enter__(self):
        """Enable use of the timer as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically stop the timer when exiting a context block."""
        self.stop()