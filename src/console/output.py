import json
from pygments import highlight, lexers, formatters
from typing import Any, Dict

def gradient_color(value, min_val=0, max_val=1, reverse=False):
    """
    Returns a smooth color gradient for a numeric value.

    Args:
        value: value to be colored
        min_val: minimum expected value (default: 0)
        max_val: maximum expected value (default: 1)
        reverse: if True — gradient from green to red;
                 if False — from red to green (default)

    Returns:
        String with ANSI color code and formatted value
    """
    # Normalize value to range [0, 1]
    v = (value - min_val) / (max_val - min_val)
    v = max(0, min(1, v))  # Clamp to [0, 1]

    if reverse:
        # Gradient: green (v=0) → red (v=1)
        r = int(255 * v)      # r increases from 0 to 255
        g = int(255 * (1 - v))  # g decreases from 255 to 0
        b = 0
    else:
        # Gradient: red (v=0) → green (v=1)
        r = int(255 * (1 - v))  # r decreases from 255 to 0
        g = int(255 * v)       # g increases from 0 to 255
        b = 0

    return f"\033[38;2;{r};{g};{b}m{value:.3f}\033[0m"


def colored_text(text, color_name='blue'):
    """
    Return text colored with the specified named color or hex-code.

    Args:
        text: text to output
        color_name: color name from the reserved color list OR hex string (#RRGGBB)

    Returns:
        String with ANSI color code and text
    """
    # Dictionary of reserved colors (RGB)
    COLORS = {
        'red':     (255, 0, 0),
        'green':   (0, 255, 0),
        'blue':    (0, 0, 255),
        'yellow':  (255, 255, 0),
        'cyan':    (0, 255, 255),
        'magenta': (255, 0, 255),
        'white':   (255, 255, 255),
        'black':   (0, 0, 0),
        'orange':  (255, 165, 0),
        'purple':  (128, 0, 128),
        'pink':    (255, 192, 203),
        'brown':   (165, 42, 42),
    }

    # if a hex code was provided (#RRGGBB or RRGGBB)
    if color_name and color_name.startswith('#'):
        hex_code = color_name[1:]
        if len(hex_code) != 6:
            raise ValueError("Hex color must be 6 digits: #RRGGBB")
        r, g, b = (int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    elif color_name and color_name.lower() in COLORS:
        r, g, b = COLORS[color_name.lower()]
    else:
        raise ValueError(f"Unknown color: {color_name}. "
                         f"Use name from {list(COLORS.keys())} or hex #RRGGBB.")

    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def print_params_json(params: dict):
    """Pretty-print JSON with syntax highlighting."""
    formatted_json = json.dumps(params, indent=4, ensure_ascii=False)
    colorful = highlight(
        formatted_json,
        lexers.JsonLexer(),
        formatters.TerminalFormatter()
    )
    print(colorful)

if __name__ == "__main__":
    print(f"IoU (normal): {gradient_color(0.3, 0, 1, reverse=False)}")      # reddish
    print(f"Dice (normal): {gradient_color(0.85, 0, 1, reverse=False)}")    # greenish

    print(f"Error (reversed): {gradient_color(0.3, 0, 1, reverse=True)}")     # greenish
    print(f"Loss (reversed): {gradient_color(0.85, 0, 1, reverse=True)}")   # reddish

    # Examples of colored_text usage
    print(colored_text("Red", "red"))
    print(colored_text("Green", "green"))
    print(colored_text("Blue", "blue"))
    print(colored_text("Yellow", "yellow"))
    print(colored_text("Hello world!", "purple"))
