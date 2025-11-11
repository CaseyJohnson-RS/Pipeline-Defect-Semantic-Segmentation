def gradient_color(value, min_val=0, max_val=1, reverse=False):
    """
    Плавный градиент для числового значения.

    Args:
        value: значение для окрашивания
        min_val: минимальное ожидаемое значение (по умолчанию 0)
        max_val: максимальное ожидаемое значение (по умолчанию 1)
        reverse: если True — градиент от зелёного к красному;
                  если False — от красного к зелёному (стандарт)

    Returns:
        Строка с ANSI‑кодом цвета и форматированным значением
    """
    # Нормализуем value в диапазон [0, 1]
    v = (value - min_val) / (max_val - min_val)
    v = max(0, min(1, v))  # ограничиваем в пределах [0, 1]

    if reverse:
        # Градиент: зелёный (v=0) → красный (v=1)
        r = int(255 * v)      # r растёт от 0 до 255
        g = int(255 * (1 - v))  # g падает от 255 до 0
        b = 0
    else:
        # Градиент: красный (v=0) → зелёный (v=1)
        r = int(255 * (1 - v))  # r падает от 255 до 0
        g = int(255 * v)       # g растёт от 0 до 255
        b = 0

    return f"\033[38;2;{r};{g};{b}m{value:.3f}\033[0m"



def colored_text(text, color_name):
    """
    Печатает текст заданным именованным цветом.

    Args:
        text: текст для вывода
        color_name: название цвета из зарезервированного списка

    Returns:
        Строка с ANSI‑кодом цвета и текстом
    """
    # Словарь зарезервированных цветов (RGB)
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

    if color_name.lower() not in COLORS:
        raise ValueError(f"Неизвестный цвет: {color_name}. Доступные: {list(COLORS.keys())}")

    r, g, b = COLORS[color_name.lower()]
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"



if __name__ == "__main__":
    print(f"IoU (normal): {gradient_color(0.3, 0, 1, reverse=False)}")      # красноватый
    print(f"Dice (normal): {gradient_color(0.85, 0, 1, reverse=False)}")    # зеленоватый

    print(f"Error (reversed): {gradient_color(0.3, 0, 1, reverse=True)}")     # зеленоватый
    print(f"Loss (reversed): {gradient_color(0.85, 0, 1, reverse=True)}")   # красноватый

    # Примеры использования colored_text
    print(colored_text("Red", "red"))
    print(colored_text("Green", "green"))
    print(colored_text("Blue", "blue"))
    print(colored_text("Yellow", "yellow"))
    print(colored_text("Hello world!", "purple"))
