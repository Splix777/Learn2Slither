import logging
from pynput import keyboard

# Set up logging
logging.basicConfig(
    filename="key_listener.log", 
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class KeyListener:
    def __init__(self) -> None:
        self.current_key: str | None = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        logging.info("KeyListener started.")

    def on_press(self, key: keyboard.Key) -> None:
        try:
            # Log key press
            logging.debug(f"Pressed key: {key} (char: {getattr(key, 'char', None)})")
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()  # Convert to lowercase for uniformity
                if char == "w":
                    self.current_key = "0"
                elif char == "s":
                    self.current_key = "1"
                elif char == "a":
                    self.current_key = "2"
                elif char == "d":
                    self.current_key = "3"
                logging.info(f"Detected movement key: {self.current_key}")
        except AttributeError as e:
            logging.warning(f"Special key pressed: {key} (Exception: {e})")

    def get_key(self) -> str | None:
        key: str | None = self.current_key
        self.current_key = None  # Clear the current key after returning it
        return key


if __name__ == "__main__":
    import time

    key_listener = KeyListener()
    logging.info("Script started. Press W, S, A, or D. Press Ctrl+C to exit.")

    try:
        while True:
            if key := key_listener.get_key():
                logging.info(f"Main loop detected key: {key}")
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    except KeyboardInterrupt:
        key_listener.listener.stop()
        logging.info("Listener stopped.")
