import os
import sys
from PIL import Image
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


input_dir = "datasets/sorted_images/Obstacle"
output_dir = "datasets/sorted_images/Obstacle_png"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        src = os.path.join(input_dir, filename)
        dst = os.path.join(
            output_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        with Image.open(src) as img:
            img.save(dst, "PNG")
