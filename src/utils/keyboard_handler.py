"""Keyboard handler module."""

from pynput import keyboard


class KeyListener:
    def __init__(self) -> None:
        self.current_key: str | None = None
        self.key_map = {
            "w": "0",
            "s": "1",
            "a": "2",
            "d": "3",
            keyboard.Key.up: "0",
            keyboard.Key.down: "1",
            keyboard.Key.left: "2",
            keyboard.Key.right: "3",
        }
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press event."""
        from contextlib import suppress

        with suppress(AttributeError):
            # Log key press
            char = key.char
            self.current_key = self.key_map.get(char)

    def get_key(self) -> str | None:
        """Get the current key."""
        key: str | None = self.current_key
        self.current_key = None
        return key
