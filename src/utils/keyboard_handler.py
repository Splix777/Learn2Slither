from pynput import keyboard


class KeyListener:
    def __init__(self) -> None:
        self.current_key: str | None = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key) -> None:
        try:
            # Handle key presses and set current_key
            self.current_key = key.char.upper()

        except AttributeError:
            # Handle special keys
            if key == keyboard.Key.up:
                self.current_key = "W"
            elif key == keyboard.Key.down:
                self.current_key = "S"
            elif key == keyboard.Key.left:
                self.current_key = "A"
            elif key == keyboard.Key.right:
                self.current_key = "D"

    def get_key(self) -> str | None:
        key: str | None = self.current_key
        self.current_key = None
        return key
