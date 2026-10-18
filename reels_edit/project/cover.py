"""Reels cover 1080x1920. Key text stays inside the 1080x1440 centre (profile-grid crop 3:4)."""
import subprocess, sys
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import render as R

src, out = sys.argv[1], sys.argv[2]
T = 43.60   # on the phone with Danil; a frame where the author's old captions are not on screen

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ss", str(T), "-frames:v", "1", "-vf",
                      "eq=contrast=1.08:saturation=1.1:gamma=0.97,curves=all='0/0 0.25/0.23 0.75/0.78 1/1'",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
frame = Image.frombuffer("RGB", (R.SW, R.SH), raw, "raw", "RGB", 0, 1)
# no old captions in this frame, so the whole picture can be used (slight zoom toward the speaker)
img = frame.resize((R.W, R.H), Image.LANCZOS, box=(40, 30, 700, 1203))
img = img.filter(ImageFilter.UnsharpMask(2.2, 60, 2)).convert("RGBA")

# darkening gradient bottom + light top for text legibility
yy = np.linspace(0, 1, R.H)[:, None]
a = np.clip((yy - 0.48) / 0.40, 0, 1) ** 1.3 * 225 + np.clip((0.16 - yy) / 0.16, 0, 1) * 110
g = Image.new("RGBA", (R.W, R.H), (8, 9, 14, 0))
g.putalpha(Image.fromarray(np.repeat(a, R.W, 1).astype(np.uint8), "L"))
img.alpha_composite(g)
img.alpha_composite(R.vignette())
d = ImageDraw.Draw(img)

def text_c(y, s, size, fill, stroke=8, x=R.W / 2, font=R.FONT_BLACK):
    f = R.font(font, size)
    b = d.textbbox((0, 0), s, font=f, stroke_width=stroke)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((x - (b[0] + b[2]) / 2, y + 10), s, font=f, fill=(0, 0, 0, 170),
                            stroke_width=stroke + 2, stroke_fill=(0, 0, 0, 170))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)))
    d.text((x - (b[0] + b[2]) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=(10, 10, 14))
    return b[3] - b[1]

# top: series label
f = R.font(R.FONT_BLACK, 46)
pill = "ЭПИЗОД 1"
b = d.textbbox((0, 0), pill, font=f)
pw, ph = b[2] - b[0] + 56, 84
px, py = (R.W - pw) / 2, 290
d.rounded_rectangle((px, py, px + pw, py + ph), radius=ph / 2, fill=R.YELLOW)
d.text((R.W / 2 - (b[0] + b[2]) / 2, py + ph / 2 - (b[1] + b[3]) / 2), pill, font=f, fill=R.DARK)
text_c(py + ph + 26, "СТРОЙКА ДЕТСКОГО ЦЕНТРА", 50, (255, 255, 255), stroke=5, font=R.FONT_XB)

# main title
def fit(s, size, maxw=980):
    while d.textbbox((0, 0), s, font=R.font(R.FONT_BLACK, size), stroke_width=8)[2] > maxw: size -= 2
    return size
size = min(fit("НОВЫЕ ОКНА", 150), fit("НА МУСОРКУ?!", 150))
y = 1225
y += text_c(y, "НОВЫЕ ОКНА", size, (255, 255, 255)) + 30
text_c(y, "НА МУСОРКУ?!", size, R.RED)
# subline (fact from the video)
f2 = R.font(R.FONT_XB, 50)
sub = "70% окон оказались разного размера"
b = d.textbbox((0, 0), sub, font=f2)
sw = b[2] - b[0] + 60
sx, sy = (R.W - sw) / 2, 1560
d.rounded_rectangle((sx, sy, sx + sw, sy + 86), radius=24, fill=(255, 255, 255, 235))
d.text((R.W / 2 - (b[0] + b[2]) / 2, sy + 43 - (b[1] + b[3]) / 2), sub, font=f2, fill=R.DARK)

# accent icons reused from the edit
trash = R.badge(R.RED, R.g_trash, 170)
img.alpha_composite(trash.rotate(-12, Image.BICUBIC, expand=True), (790, 1010))
warn = R.warn_sprite(150)
img.alpha_composite(warn.rotate(10, Image.BICUBIC, expand=True), (90, 1035))

img.convert("RGB").save(out, quality=95)
