from PIL import Image

def create_color_block(color: tuple, filename: str, size: int = 1) -> None:
    """
    Create a solid color block PNG image.

    Args:
        color (tuple): RGB color as (R, G, B), e.g., (255, 0, 0) for red.
        filename (str): The name of the output file (e.g., "red_block.png").
        size (int): The width and height of the block in pixels. Default is 20.
    """
    # Create a new image with the given color
    image = Image.new("RGB", (size, size), color)
    
    # Save the image to a file
    image.save(filename)
    print(f"Saved: {filename}")

# Color mapping
colors = {
    "red": (255, 0, 0),            # Red Apple/Dark/Light
    "green": (0, 255, 0),          # Green Apple/Dark/Light
    "blue": (0, 0, 255),           # Snake Head/Dark/Light
    "yellow": (255, 255, 0),       # Snake Body/Dark/Light
    "cyan": (0, 255, 255),         # Wall Dark
    "magenta": (255, 0, 255),      # Wall Light
    "white": (255, 255, 255),      # Empty Light
    "black": (0, 0, 0),            # Empty Dark
}

# Files needed
files_needed = [
    "empty_dark.png",
    "empty_light.png",
    # "green_apple_dark.png",
    # "green_apple_light.png",
    # "red_apple_dark.png",
    # "red_apple_light.png",
    # "snake_body_dark.png",
    # "snake_body_light.png",
    # "snake_head_dark.png",
    # "snake_head_light.png",
    # "wall_dark.png",
    # "wall_light.png",
]

# Mapping file names to colors
file_color_mapping = {
    "empty_dark": "black",
    "empty_light": "white",
    "green_apple_dark": "green",
    "green_apple_light": "green",
    "red_apple_dark": "red",
    "red_apple_light": "red",
    "snake_body_dark": "yellow",
    "snake_body_light": "yellow",
    "snake_head_dark": "blue",
    "snake_head_light": "blue",
    "wall_dark": "cyan",
    "wall_light": "magenta",
}

# Generate all files
for file_name in files_needed:
    base_name = file_name.split(".")[0]  # Remove the extension
    color_name = file_color_mapping[base_name]  # Get the color key
    color = colors[color_name]  # Look up the RGB tuple
    create_color_block(color, file_name)
