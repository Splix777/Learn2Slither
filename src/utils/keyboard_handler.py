from pynput import keyboard


class KeyListener:
    def __init__(self) -> None:
        self.current_key: str | None = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()


    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        from contextlib import suppress

        with suppress(AttributeError):
            # Log key press
            char = key.char.lower()
            key_map = {"w": "1", "s": "2", "a": "3", "d": "4"}
            self.current_key = key_map.get(char)

    def get_key(self) -> str | None:
        key: str | None = self.current_key
        self.current_key = None
        return key
