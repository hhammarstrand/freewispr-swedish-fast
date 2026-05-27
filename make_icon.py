"""Generate icon.ico for the freewispr .exe build."""
import os

from PIL import Image, ImageDraw


def make_icon(size=256):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Purple rounded background
    draw.rounded_rectangle([0, 0, size, size], radius=size // 5, fill="#7c5cfc")
    cx = size // 2
    s = size / 64  # scale factor
    # Mic body
    draw.rounded_rectangle(
        [cx - 9*s, 12*s, cx + 9*s, 36*s],
        radius=9*s, fill="white"
    )
    # Mic arc
    draw.arc(
        [cx - 16*s, 26*s, cx + 16*s, 50*s],
        start=0, end=180, fill="white", width=int(3*s)
    )
    # Stand line + base
    draw.line([cx, 50*s, cx, 58*s], fill="white", width=int(3*s))
    draw.line([cx - 8*s, 58*s, cx + 8*s, 58*s], fill="white", width=int(3*s))
    return img

sizes = [16, 32, 48, 64, 128, 256]
frames = [make_icon(s) for s in sizes]

os.makedirs("assets", exist_ok=True)
frames[0].save(
    "assets/icon.ico",
    format="ICO",
    sizes=[(s, s) for s in sizes],
    append_images=frames[1:]
)
print("assets/icon.ico created.")
