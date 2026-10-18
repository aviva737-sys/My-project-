"""Reels re-edit renderer.

Pipeline per output frame (30 fps, 1080x1920):
  source frame (graded by ffmpeg) -> virtual camera crop (tracking + zoom punches,
  always above y=925 so the author's burned-in subtitles never enter the frame)
  -> motion graphics layer -> kinetic captions -> encoder.

Usage:
  render.py full    <src> <out.mp4> [f0 f1]   final picture
  render.py overlay <src> <out.mov> [f0 f1]   graphics+captions only, with alpha (project layer)
  render.py base    <src> <out.mp4> [f0 f1]   reframed/graded picture only (project layer)
  render.py events  <out.json>                 timeline of sound-effect cues for audio.py
"""
import json, math, os, subprocess, sys
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
SW, SH = 720, 1280
DURATION = 58.86
FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_XB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WHITE = (255, 255, 255)
YELLOW = (255, 210, 63)
RED = (255, 77, 77)
GREEN = (52, 211, 120)
BLUE = (64, 156, 255)
ORANGE = (255, 146, 43)
DARK = (14, 16, 22)

CUTS = [4.10, 22.33, 29.81, 43.76, 52.08]
SAFE_BOTTOM = 925           # never sample source rows below this (old captions start ~950)

# ----------------------------------------------------------------- easing helpers
def clamp(v, a=0.0, b=1.0): return a if v < a else b if v > b else v
def ease_out_cubic(p): p = clamp(p); return 1 - (1 - p) ** 3
def ease_in_out(p): p = clamp(p); return p * p * (3 - 2 * p)
def ease_out_back(p, s=1.9):
    p = clamp(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2
def lerp(a, b, p): return a + (b - a) * p

# ----------------------------------------------------------------- camera
# Head x-position in source pixels, sampled once per second from the footage.
HEAD_X = {4: 368, 5: 416, 6: 384, 7: 400, 8: 480, 9: 448, 10: 464, 11: 432, 12: 464, 13: 384,
          14: 512, 15: 464, 16: 448, 17: 400, 18: 464, 19: 560, 20: 416, 21: 400, 22: 432,
          23: 400, 24: 500, 25: 520, 26: 400, 27: 496, 28: 512, 29: 464, 30: 480, 31: 544,
          32: 600, 33: 620, 34: 560, 35: 540, 36: 384, 37: 352, 38: 384, 39: 520, 40: 320,
          41: 448, 42: 400, 43: 470, 44: 400, 45: 400, 46: 432, 47: 560, 48: 400, 49: 432,
          50: 300, 51: 448, 52: 368, 53: 368, 54: 300, 55: 384, 56: 330, 57: 368, 58: 352, 59: 352}
BASE_W, BASE_H = 518, 920   # 9:16 window that fits above the old caption band

def shot_of(t):
    b = [0.0] + CUTS + [99]
    for i in range(len(b) - 1):
        if b[i] <= t < b[i + 1]: return b[i], b[i + 1]
    return CUTS[-1], 99

def head_x(t):
    s0, s1 = shot_of(t)
    # gaussian-weighted average of samples inside the same shot -> smooth virtual pan
    num = den = 0.0
    for k, v in HEAD_X.items():
        if not (s0 - 0.5 <= k <= s1 + 0.5): continue
        w = math.exp(-((k - t) ** 2) / (2 * 0.9 ** 2))
        num += w * v; den += w
    x = num / den if den else 400
    return clamp(x, BASE_W / 2, SW - BASE_W / 2)

# zoom punches: (start, peak_reached, release_start, end, amount)
PUNCHES = [
    (2.70, 2.95, 3.55, 3.95, 0.07),
    (6.75, 7.00, 7.75, 8.15, 0.09),
    (12.30, 12.55, 13.05, 13.35, 0.08),
    (15.55, 15.80, 16.45, 16.85, 0.08),
    (21.10, 21.35, 22.05, 22.33, 0.07),
    (26.20, 26.45, 27.40, 27.75, 0.06),
    (37.85, 38.05, 38.50, 38.85, 0.12),
    (43.25, 43.45, 43.62, 43.76, 0.08),
    (45.00, 45.25, 46.20, 46.55, 0.09),
    (53.95, 54.15, 54.40, 54.70, 0.08),
    (56.90, 58.20, 58.86, 59.00, 0.10),
]
SHAKES = [(3.40, 0.35, 5.0), (37.87, 0.40, 6.0), (43.30, 0.35, 5.0), (53.98, 0.25, 3.0)]

def zoom_at(t):
    s0, s1 = shot_of(t)
    z = 1.0 + 0.035 * clamp((t - s0) / max(1.0, s1 - s0))      # slow push-in inside every shot
    for a, b, c, d, amt in PUNCHES:
        if a <= t <= d:
            p = ease_in_out((t - a) / (b - a)) if t < b else 1.0 if t < c else 1 - ease_in_out((t - c) / (d - c))
            z *= 1 + amt * p
    for c in CUTS:                                                 # impact zoom on every cut
        if c <= t < c + 0.38:
            z *= 1 + 0.10 * (1 - ease_out_cubic((t - c) / 0.38))
    return z

def shake_at(t):
    dx = dy = 0.0
    for t0, dur, amp in SHAKES:
        if t0 <= t < t0 + dur:
            k = 1 - (t - t0) / dur
            dx += amp * k * math.sin((t - t0) * 95)
            dy += amp * k * math.cos((t - t0) * 77)
    return dx, dy

def camera_box(t):
    if t < CUTS[0]:   # B&W hook: crop inside the viewfinder graphics (keeps focus brackets)
        bw, bh, bx, by = 413, 735, 233, 185
        fx, fy = 440, 520
    else:
        cx = head_x(t)
        bw, bh, bx, by = BASE_W, BASE_H, cx - BASE_W / 2, 0
        fx = cx
        fy = 560 if t >= 53.9 else (230 if t >= CUTS[-1] else 430)
    z = zoom_at(t)
    w, h = bw / z, bh / z
    x0 = fx - (fx - bx) / z
    y0 = fy - (fy - by) / z
    dx, dy = shake_at(t)
    x0 = clamp(x0 + dx, 0, SW - w)
    y0 = clamp(y0 + dy, 0, SAFE_BOTTOM - h)
    return x0, y0, x0 + w, y0 + h

# ----------------------------------------------------------------- captions
KEY = {"2", "70%", "стандартные", "крайних", "размеру", "гипсокартоном", "манипулятор", "замерять",
       "центральное", "заказ", "гипсокартон", "ширину", "высоту", "рандомных", "размера", "высоте",
       "ширине", "переформировать", "цена", "Данилу", "окон", "удостовериться", "пишите", "избежать"}
NEG = {"мусорку", "попали", "плохо", "ошибках", "ошибки", "не"}
BIG = {"2": 1.45, "70%": 1.45, "переформировать": 0.86}
# pages whose speaker is in close-up -> captions go above the head
TOP_PAGES_FROM = [(37.1, 40.0), (53.9, 99)]
CAP_BOTTOM_Y, CAP_TOP_Y = 1300, 420
FS = 92
MAXW = 940

def load_pages():
    raw = json.load(open(os.path.join(HERE, "words.json"), encoding="utf-8"))
    pages = []
    for i, p in enumerate(raw):
        words = [[w, t] for w, t in p if w != "END"]
        end = next((t for w, t in p if w == "END"), None)
        for j in range(1, len(words)):                      # keep onsets strictly increasing
            words[j][1] = max(words[j][1], words[j - 1][1] + 0.08)
        pages.append({"words": words, "end": end})
    for i, p in enumerate(pages):
        nxt = pages[i + 1]["words"][0][1] if i + 1 < len(pages) else DURATION
        p["end"] = min(p["end"] + 0.12, nxt) if p["end"] else nxt
    return pages

@lru_cache(None)
def font(path, size): return ImageFont.truetype(path, size)

def word_size(text, scale=1.0):
    f = font(FONT_BLACK, int(FS * scale))
    b = f.getbbox(text)
    return b[2] - b[0], f

CHUNKS = """я звоню Данилу | и говорю | Данил | новые окна уходят | на мусорку | мы попали |
сначала мы думали | что все окна | здесь стандартные | кроме 2 крайних окон | они немножечко |
отличаются по размеру | но так как | внутренняя сторона | вся была зашита | гипсокартоном |
мы не могли | удостовериться | в этой информации | мы вызвали | манипулятор | и начали замерять окна |
мы замерили | 2 крайних | центральное | по ним мы | сформировали | наш заказ | но после того как |
мастера здесь | расшили гипсокартон | я замерил | несколько окон | я беру за пример | это окно |
замеряю его | ширину высоту | замеряю ещё | несколько окон | рандомных | и понимаю | что всё плохо |
все окна | разного размера | я звоню Данилу | и говорю | Данил | новые окна уходят | на мусорку |
мы попали | оказывается | что они наверное | 70% окон | разные по высоте | разные по ширине |
и мы успеваем | полностью | переформировать | заказ | на примере | этого объекта | мы учимся |
на своих ошибках | и стараемся их | максимально избежать | как думаете | какова цена |
была бы этой ошибки | пишите"""

def build_chunks():
    space = FS * 0.30
    flat = []
    for p in load_pages():
        for k, (w, t) in enumerate(p["words"]):
            flat.append((w, t, p["end"] if k == len(p["words"]) - 1 else None))
    chunks, i = [], 0
    for phrase in CHUNKS.replace("\n", " ").split("|"):
        ws = phrase.split()
        part = flat[i:i + len(ws)]
        assert [w for w, _, _ in part] == ws, (ws, part)
        i += len(ws)
        chunks.append({"words": [[w, t] for w, t, _ in part], "page_end": part[-1][2]})
    assert i == len(flat)
    for k, c in enumerate(chunks):
        start = c["words"][0][1]
        nxt = chunks[k + 1]["words"][0][1] if k + 1 < len(chunks) else DURATION
        c["start"] = start
        c["end"] = min(nxt, c["page_end"]) if c["page_end"] else nxt
        c["end"] = min(nxt, max(c["end"], c["words"][-1][1] + 0.5))   # never flash shorter than 0.5 s
        top = any(a <= start < b for a, b in TOP_PAGES_FROM)
        c["cy"] = CAP_TOP_Y if top else CAP_BOTTOM_Y
        place_words(c, space)
    return chunks

def layout_lines(words, space):
    lines, cur, cw = [], [], 0
    for w in words:
        ww = word_size(w.upper(), BIG.get(w, 1))[0]
        if cur and cw + space + ww > MAXW:
            lines.append(cur); cur, cw = [w], ww
        else:
            cw = cw + (space if cur else 0) + ww; cur.append(w)
    if cur: lines.append(cur)
    return lines

def place_words(c, space):
    lines = layout_lines([w for w, _ in c["words"]], space)
    lh = FS * 1.12
    total = lh * len(lines)
    y = c["cy"] - total / 2 + lh / 2
    k = 0
    c["placed"] = []
    for line in lines:
        widths = [word_size(w.upper(), BIG.get(w, 1))[0] for w in line]
        x = W / 2 - (sum(widths) + space * (len(line) - 1)) / 2
        for w, ww in zip(line, widths):
            t = c["words"][k][1]
            c["placed"].append((w, t, x + ww / 2, y))
            x += ww + space; k += 1
        y += lh

@lru_cache(None)
def word_sprite(text, color, scale):
    """White/colored word with dark stroke and soft drop shadow, as RGBA."""
    f = font(FONT_BLACK, int(FS * scale))
    s = text.upper()
    b = f.getbbox(s, stroke_width=7)
    pad = 34
    w, h = b[2] - b[0] + pad * 2, b[3] - b[1] + pad * 2
    org = (pad - b[0], pad - b[1])
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).text((org[0], org[1] + 7), s, font=f, fill=150, stroke_width=9, stroke_fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.putalpha(shadow)
    d = ImageDraw.Draw(img)
    d.text(org, s, font=f, fill=color + (255,), stroke_width=7, stroke_fill=(10, 10, 14, 255))
    # visual centre of the glyphs (for anchoring) relative to the sprite
    cap = f.getbbox("Н")
    return img, (org[0] + (b[0] + b[2]) / 2, org[1] + (cap[1] + cap[3]) / 2)

def word_color(w):
    if w in NEG: return RED
    if w in KEY or any(ch.isdigit() for ch in w): return YELLOW
    return WHITE

def draw_captions(canvas, t, chunks):
    for c in chunks:
        if not (c["start"] - 0.01 <= t < c["end"]): continue
        out = clamp((c["end"] - t) / 0.10)           # quick exit fade
        for idx, (w, onset, cx, cy) in enumerate(c["placed"]):
            dt = t - onset
            if dt < 0: continue
            scale_word = BIG.get(w, 1.0)
            nxt = c["placed"][idx + 1][1] if idx + 1 < len(c["placed"]) else c["end"]
            active = onset <= t < nxt
            p = dt / 0.22
            s = lerp(0.45, 1.0, ease_out_back(p)) if p < 1 else 1.0
            if active and p >= 1:
                s = 1.0 + 0.06 * (1 - ease_out_cubic((dt - 0.22) / 0.25))
            s *= lerp(0.94, 1.0, out)
            a = clamp(dt / 0.07) * out
            dy = (1 - ease_out_cubic(p)) * 22
            spr, (ax, ay) = word_sprite(w, word_color(w), scale_word)
            paste_scaled(canvas, spr, cx, cy + dy, s, a, anchor=(ax, ay))

def paste_scaled(canvas, spr, cx, cy, s, a, anchor=None, rot=0):
    if a <= 0.003 or s <= 0.01: return
    w, h = spr.size
    ax, ay = anchor if anchor else (w / 2, h / 2)
    if abs(s - 1) > 1e-3:
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        spr = spr.resize((nw, nh), Image.BICUBIC)
        ax, ay = ax * nw / w, ay * nh / h
    if rot:
        spr = spr.rotate(rot, Image.BICUBIC, expand=True)
        ax, ay = spr.size[0] / 2, spr.size[1] / 2
    if a < 0.999:
        al = spr.getchannel("A").point(lambda v: int(v * a))
        spr = spr.copy(); spr.putalpha(al)
    x, y = int(round(cx - ax)), int(round(cy - ay))
    canvas.alpha_composite(spr, (max(0, x), max(0, y)),
                           (max(0, -x), max(0, -y)))

# ----------------------------------------------------------------- icon drawing (2x supersampled)
SS = 2

def new_layer(w, h): return Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
def finish(img): return img.resize((img.size[0] // SS, img.size[1] // SS), Image.LANCZOS)

def with_shadow(img, blur=14, off=8, alpha=120):
    a = img.getchannel("A").point(lambda v: v * alpha // 255)
    pad = blur * 3
    sh = Image.new("RGBA", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0, 0))
    m = Image.new("L", sh.size, 0); m.paste(a, (pad, pad + off))
    m = m.filter(ImageFilter.GaussianBlur(blur))
    sh.putalpha(m)
    sh.alpha_composite(img, (pad, pad))
    return sh

def P(pts, s, ox=0, oy=0): return [(ox + x * s, oy + y * s) for x, y in pts]

def rot_pts(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]

def badge(color, glyph, size=190, glyph_arg=None):
    """Round colored badge with a white pictogram."""
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, S - 1, S - 1), fill=color + (255,))
    # subtle top highlight
    hl = Image.new("L", (S, S), 0)
    ImageDraw.Draw(hl).ellipse((S * 0.08, S * 0.02, S * 0.92, S * 0.6), fill=40)
    img.alpha_composite(Image.merge("RGBA", (hl.point(lambda v: 255),) * 3 + (hl,)))
    glyph(d, S, glyph_arg)
    d.ellipse((0, 0, S - 1, S - 1), outline=(255, 255, 255, 230), width=int(S * 0.035))
    return with_shadow(finish(img))

def g_phone(d, S, _):
    # handset drawn as a thick rounded arc with two ear/mouth pieces
    c = S / 2
    w = int(S * 0.13)
    d.arc((S * 0.27, S * 0.27, S * 0.73, S * 0.73), 110, 250, fill=WHITE, width=w)
    for ang in (110, 250):
        a = math.radians(ang)
        x, y = c + S * 0.23 * math.cos(a), c + S * 0.23 * math.sin(a)
        pts = rot_pts([(x - S * 0.09, y - S * 0.05), (x + S * 0.09, y - S * 0.05),
                       (x + S * 0.09, y + S * 0.05), (x - S * 0.09, y + S * 0.05)], a + math.pi / 2, x, y)
        d.polygon(pts, fill=WHITE)
    # sound waves
    for r in (0.16, 0.25):
        d.arc((c + S * 0.02 - S * r, c - S * r, c + S * 0.02 + S * r, c + S * r), -50, 50, fill=WHITE, width=int(S * 0.045))

def g_trash(d, S, _):
    u = S / 100
    d.rounded_rectangle((30 * u, 30 * u, 70 * u, 37 * u), radius=3 * u, fill=WHITE)
    d.rounded_rectangle((43 * u, 24 * u, 57 * u, 31 * u), radius=3 * u, fill=WHITE)
    d.polygon(P([(33, 41), (67, 41), (63, 76), (37, 76)], u), fill=WHITE)
    for x in (43, 50, 57):
        d.line(P([(x, 47), (x - (x - 50) * 0.1, 70)], u), fill=RED + (255,), width=int(3.5 * u))

def g_text(d, S, arg):
    txt, fsz, col = arg
    f = font(FONT_BLACK, int(S * fsz))
    b = d.textbbox((0, 0), txt, font=f)
    d.text((S / 2 - (b[0] + b[2]) / 2, S / 2 - (b[1] + b[3]) / 2), txt, font=f, fill=col)

def g_magnifier(d, S, _):
    u = S / 100
    d.ellipse((24 * u, 22 * u, 66 * u, 64 * u), outline=WHITE, width=int(8 * u))
    d.line(P([(60, 58), (77, 76)], u), fill=WHITE, width=int(11 * u))
    f = font(FONT_BLACK, int(28 * u))
    b = d.textbbox((0, 0), "?", font=f)
    d.text((45 * u - (b[0] + b[2]) / 2, 43 * u - (b[1] + b[3]) / 2), "?", font=f, fill=WHITE)

def g_hammer(d, S, _):
    u = S / 100
    ang = math.radians(-40)
    handle = rot_pts(P([(46, 38), (54, 38), (54, 82), (46, 82)], u), ang, 50 * u, 50 * u)
    head = rot_pts(P([(30, 24), (70, 24), (70, 38), (30, 38)], u), ang, 50 * u, 50 * u)
    claw = rot_pts(P([(70, 24), (80, 20), (76, 31), (70, 38)], u), ang, 50 * u, 50 * u)
    d.polygon(handle, fill=WHITE); d.polygon(head, fill=WHITE); d.polygon(claw, fill=WHITE)

def g_check(d, S, _):
    u = S / 100
    d.line(P([(27, 52), (43, 67), (74, 34)], u), fill=WHITE, width=int(11 * u), joint="curve")

def g_cross(d, S, _):
    u = S / 100
    d.line(P([(32, 32), (68, 68)], u), fill=WHITE, width=int(12 * u))
    d.line(P([(68, 32), (32, 68)], u), fill=WHITE, width=int(12 * u))

def g_arrows(d, S, vertical):
    u = S / 100
    if vertical:
        d.line(P([(50, 26), (50, 74)], u), fill=WHITE, width=int(8 * u))
        d.polygon(P([(36, 34), (64, 34), (50, 18)], u), fill=WHITE)
        d.polygon(P([(36, 66), (64, 66), (50, 82)], u), fill=WHITE)
    else:
        d.line(P([(26, 50), (74, 50)], u), fill=WHITE, width=int(8 * u))
        d.polygon(P([(34, 36), (34, 64), (18, 50)], u), fill=WHITE)
        d.polygon(P([(66, 36), (66, 64), (82, 50)], u), fill=WHITE)

def g_building(d, S, _):
    u = S / 100
    d.rectangle((30 * u, 26 * u, 70 * u, 78 * u), fill=WHITE)
    for r in range(4):
        for c in range(3):
            x, y = 36 + c * 11, 32 + r * 10
            d.rectangle((x * u, y * u, (x + 6) * u, (y + 6) * u), fill=BLUE + (255,))
    d.rectangle((45 * u, 70 * u, 55 * u, 78 * u), fill=BLUE + (255,))

def g_shield(d, S, _):
    u = S / 100
    d.polygon(P([(50, 20), (76, 30), (73, 58), (50, 80), (27, 58), (24, 30)], u), fill=WHITE)
    d.line(P([(37, 50), (47, 60), (64, 40)], u), fill=GREEN + (255,), width=int(8 * u), joint="curve")

def g_bubble(d, S, _):
    u = S / 100
    d.rounded_rectangle((22 * u, 26 * u, 78 * u, 66 * u), radius=12 * u, fill=WHITE)
    d.polygon(P([(34, 64), (48, 64), (32, 80)], u), fill=WHITE)
    for x in (37, 50, 63):
        d.ellipse(((x - 4.5) * u, 41.5 * u, (x + 4.5) * u, 50.5 * u), fill=BLUE + (255,))

def g_doc(d, S, checked):
    u = S / 100
    d.rounded_rectangle((30 * u, 22 * u, 70 * u, 80 * u), radius=6 * u, fill=WHITE)
    d.rounded_rectangle((40 * u, 17 * u, 60 * u, 28 * u), radius=4 * u, fill=(220, 225, 235, 255))
    for i, y in enumerate((38, 48, 58)):
        d.rounded_rectangle((37 * u, y * u, (63 - i * 6) * u, (y + 4) * u), radius=2 * u, fill=(150, 160, 175, 255))
    if checked:
        d.ellipse((52 * u, 58 * u, 80 * u, 86 * u), fill=GREEN + (255,), outline=WHITE, width=int(3 * u))
        d.line(P([(59, 72), (64, 77), (73, 66)], u), fill=WHITE, width=int(4 * u), joint="curve")

def g_refresh(d, S, _):
    u = S / 100
    w = int(8 * u)
    d.arc((26 * u, 26 * u, 74 * u, 74 * u), 200, 340, fill=WHITE, width=w)
    d.arc((26 * u, 26 * u, 74 * u, 74 * u), 20, 160, fill=WHITE, width=w)
    d.polygon(P([(72, 32), (80, 50), (62, 46)], u), fill=WHITE)
    d.polygon(P([(28, 68), (20, 50), (38, 54)], u), fill=WHITE)

def g_ruble(d, S, _):
    g_text(d, S, ("₽", 0.55, (120, 72, 0, 255)))

# ---- larger infographics (drawn per frame where animated)
def glass_panel(canvas, box, radius=34, tint=(12, 14, 20), alpha=150, a=1.0):
    """Frosted-glass card: blurs the footage behind it."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0 or a <= 0: return
    reg = canvas.crop((x0, y0, x1, y1)).convert("RGB").filter(ImageFilter.GaussianBlur(16))
    over = Image.new("RGB", reg.size, tint)
    reg = Image.blend(reg, over, alpha / 255)
    m = Image.new("L", reg.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, reg.size[0] - 1, reg.size[1] - 1), radius=radius, fill=int(255 * a))
    canvas.paste(reg.convert("RGBA"), (x0, y0), m)
    ol = Image.new("RGBA", reg.size, (0, 0, 0, 0))
    ImageDraw.Draw(ol).rounded_rectangle((1, 1, reg.size[0] - 2, reg.size[1] - 2), radius=radius,
                                         outline=(255, 255, 255, int(70 * a)), width=2)
    canvas.alpha_composite(ol, (x0, y0))

def window_icon(w, h, frame=WHITE, glass=(150, 200, 255), status=None, ring=None):
    img = new_layer(w, h)
    d = ImageDraw.Draw(img)
    u = SS
    lw = max(4, int(min(w, h) * 0.075)) * u
    d.rounded_rectangle((0, 0, w * u - 1, h * u - 1), radius=8 * u, fill=frame + (255,))
    d.rectangle((lw, lw, w * u - lw, h * u - lw), fill=glass + (235,))
    d.line((w * u / 2, lw, w * u / 2, h * u - lw), fill=frame + (255,), width=int(lw * 0.7))
    d.line((lw, h * u * 0.42, w * u - lw, h * u * 0.42), fill=frame + (255,), width=int(lw * 0.7))
    # glare
    d.polygon([(lw * 1.6, h * u * 0.8), (w * u * 0.30, lw * 1.6), (w * u * 0.40, lw * 1.6), (lw * 1.6, h * u * 0.98)],
              fill=(255, 255, 255, 60))
    return finish(img)

def mark_sprite(kind, size=64):
    col = GREEN if kind == "ok" else RED
    img = badge(col, g_check if kind == "ok" else g_cross, size)
    return img

def draw_windows_row(canvas, t, ev):
    """Row of window pictograms; mode decides which ones get highlighted/checked/resized."""
    t0, t1 = ev["t0"], ev["t1"]
    a_in = ease_out_cubic((t - t0) / 0.25)
    a_out = clamp((t1 - t) / 0.2)
    a = a_in * a_out
    cx, cy = ev["pos"]
    n = 5
    ww, wh, gap = 92, 128, 22
    total = n * ww + (n - 1) * gap
    glass_panel(canvas, (cx - total / 2 - 36, cy - wh / 2 - 40, cx + total / 2 + 36, cy + wh / 2 + 52), a=a)
    mode = ev["mode"]
    for i in range(n):
        st = t0 + 0.06 * i
        p = (t - st) / 0.3
        if p <= 0: continue
        s = ease_out_back(p) if p < 1 else 1.0
        sx = sy = 1.0
        frame, glassc = WHITE, (150, 200, 255)
        edge = i in (0, n - 1)
        mark = None
        if mode == "standard":
            hl = ev["hl"]
            if edge and t >= hl:
                q = ease_out_back((t - hl) / 0.3)
                frame = ORANGE
                sy = 1 + 0.10 * q if i == 0 else 1 - 0.08 * q
                sx = 1 - 0.08 * q if i == 0 else 1 + 0.10 * q
                if t >= ev["diff"]:   # "отличаются по размеру": breathing size difference
                    k = math.sin((t - ev["diff"]) * 7) * 0.05
                    sx += k; sy -= k
        elif mode == "measured":
            if edge and t >= ev["m1"]: frame, mark = GREEN, ("ok", ev["m1"] + 0.06 * (i > 0))
            if i == 2 and t >= ev["m2"]: frame, mark = GREEN, ("ok", ev["m2"])
        elif mode == "random":
            sizes = [(0.8, 1.15), (1.1, 0.85), (0.9, 1.0), (1.2, 1.1), (0.75, 0.9)]
            sx, sy = sizes[i]
            frame = ORANGE if ev.get("warn") and t >= ev["warn"] else WHITE
        img = window_icon(int(ww * sx), int(wh * sy), frame, glassc)
        x = cx - total / 2 + i * (ww + gap) + ww / 2
        paste_scaled(canvas, img, x, cy, s, a)
        if mark:
            kind, mt = mark
            q = (t - mt) / 0.25
            if q > 0:
                paste_scaled(canvas, mark_sprite(kind), x + ww * 0.36, cy - wh * 0.45,
                             ease_out_back(q) if q < 1 else 1, a)

def draw_drywall(canvas, t, ev):
    """Window with a gypsum board sliding over it (cover) or off it (reveal)."""
    t0, t1 = ev["t0"], ev["t1"]
    a = ease_out_cubic((t - t0) / 0.25) * clamp((t1 - t) / 0.2)
    cx, cy = ev["pos"]
    ww, wh = 200, 250
    glass_panel(canvas, (cx - 150, cy - 165, cx + 150, cy + 165), a=a)
    win = window_icon(ww, wh)
    paste_scaled(canvas, win, cx, cy, lerp(0.7, 1, ease_out_back((t - t0) / 0.3)), a)
    p = ease_in_out((t - ev["slide"]) / 0.45)
    cover = p if ev["mode"] == "cover" else 1 - p
    if cover <= 0.001: return
    bw, bh = ww + 30, wh + 30
    board = new_layer(bw, bh)
    d = ImageDraw.Draw(board)
    d.rounded_rectangle((0, 0, bw * SS - 1, bh * SS - 1), radius=6 * SS, fill=(222, 222, 216, 255),
                        outline=(180, 180, 172, 255), width=3 * SS)
    for yy in (0.12, 0.5, 0.88):
        for xx in (0.1, 0.9):
            d.ellipse(((bw * xx - 4) * SS, (bh * yy - 4) * SS, (bw * xx + 4) * SS, (bh * yy + 4) * SS), fill=(120, 120, 115, 255))
    board = finish(board)
    # slide from the right; clip to the panel
    off = (1 - cover) * (bw + 80)
    tmp = Image.new("RGBA", (300, 330), (0, 0, 0, 0))
    tmp.alpha_composite(board, (int(150 - bw / 2 + off), int(165 - bh / 2)) if off < 300 else (0, 0))
    if off >= 300: return
    al = tmp.getchannel("A").point(lambda v: int(v * a))
    tmp.putalpha(al)
    canvas.alpha_composite(tmp, (int(cx - 150), int(cy - 165)))

def draw_tape(canvas, t, ev):
    """Tape measure whose blade extends with tick marks."""
    t0, t1 = ev["t0"], ev["t1"]
    a = clamp((t - t0) / 0.12) * clamp((t1 - t) / 0.2)
    s = ease_out_back((t - t0) / 0.3) if t - t0 < 0.3 else 1
    cx, cy = ev["pos"]
    ext = ease_out_cubic((t - t0 - 0.15) / 0.6) * 300
    dirn = ev.get("dir", 1)
    Wd, Hd = 520, 220
    img = new_layer(Wd, Hd)
    d = ImageDraw.Draw(img)
    u = SS
    bx = 40 if dirn > 0 else Wd - 200
    # blade
    if ext > 2:
        x0 = bx + 150 if dirn > 0 else bx + 10 - ext
        x1 = x0 + ext
        d.rectangle((x0 * u, 128 * u, x1 * u, 156 * u), fill=(255, 214, 10, 255), outline=(40, 40, 40, 255), width=2 * u)
        k = 0
        xx = bx + 150 if dirn > 0 else bx + 10
        while k * 12 < ext:
            px = xx + (k * 12 if dirn > 0 else -k * 12)
            L = 14 if k % 5 == 0 else 7
            d.line((px * u, 128 * u, px * u, (128 + L) * u), fill=(30, 30, 30, 255), width=2 * u)
            k += 1
        hook_x = x1 if dirn > 0 else x0
        d.rectangle(((hook_x - 4) * u, 118 * u, (hook_x + 4) * u, 166 * u), fill=(60, 60, 60, 255))
    # case
    d.rounded_rectangle((bx * u, 50 * u, (bx + 160) * u, 190 * u), radius=30 * u, fill=(255, 196, 0, 255),
                        outline=(30, 30, 30, 255), width=5 * u)
    d.ellipse(((bx + 40) * u, 80 * u, (bx + 120) * u, 160 * u), fill=(40, 40, 40, 255))
    d.ellipse(((bx + 62) * u, 102 * u, (bx + 98) * u, 138 * u), fill=(255, 196, 0, 255))
    img = with_shadow(finish(img))
    paste_scaled(canvas, img, cx, cy, s, a)

def draw_crane(canvas, t, ev):
    """Manipulator truck: boom lifts up and hook swings."""
    t0, t1 = ev["t0"], ev["t1"]
    a = clamp((t - t0) / 0.12) * clamp((t1 - t) / 0.2)
    cx, cy = ev["pos"]
    glass_panel(canvas, (cx - 230, cy - 170, cx + 230, cy + 150), a=a)
    p = ease_out_back((t - t0) / 0.35) if t - t0 < 0.35 else 1
    lift = ease_in_out((t - t0 - 0.1) / 0.6)
    Wd, Hd = 440, 300
    img = new_layer(Wd, Hd)
    d = ImageDraw.Draw(img)
    u = SS
    Y = YELLOW + (255,)
    # chassis + cab
    d.rounded_rectangle((40 * u, 200 * u, 400 * u, 232 * u), radius=8 * u, fill=WHITE)
    d.rounded_rectangle((300 * u, 140 * u, 400 * u, 210 * u), radius=14 * u, fill=WHITE)
    d.rectangle((320 * u, 154 * u, 380 * u, 184 * u), fill=BLUE + (255,))
    d.rectangle((60 * u, 176 * u, 290 * u, 204 * u), fill=(210, 214, 222, 255))
    for x in (95, 175, 350):
        d.ellipse(((x - 24) * u, 214 * u, (x + 24) * u, 262 * u), fill=(30, 32, 40, 255), outline=WHITE, width=5 * u)
    # boom: base at (270,176), angle from 8deg to 38deg
    ang = math.radians(-lerp(6, 40, lift))
    bx, by = 270, 172
    L = 230
    ex, ey = bx - L * math.cos(ang), by + L * math.sin(ang)
    d.line((bx * u, by * u, ex * u, ey * u), fill=Y, width=18 * u)
    d.ellipse(((bx - 14) * u, (by - 14) * u, (bx + 14) * u, (by + 14) * u), fill=Y)
    # cable and hook with swing
    sw = math.sin((t - t0) * 6) * 10 * (1 - clamp((t - t0) / 1.2))
    hx, hy = ex + sw, ey + 60
    d.line((ex * u, ey * u, hx * u, hy * u), fill=WHITE, width=3 * u)
    d.arc(((hx - 12) * u, (hy - 2) * u, (hx + 12) * u, (hy + 22) * u), -10, 200, fill=WHITE, width=5 * u)
    img = finish(img)
    paste_scaled(canvas, img, cx, cy - 10, p, a)

def draw_dimensions(canvas, t, ev):
    """Blueprint-style window with width and height dimension lines."""
    t0, t1 = ev["t0"], ev["t1"]
    a = ease_out_cubic((t - t0) / 0.25) * clamp((t1 - t) / 0.2)
    cx, cy = ev["pos"]
    glass_panel(canvas, (cx - 200, cy - 180, cx + 200, cy + 180), tint=(10, 40, 90), alpha=165, a=a)
    win = window_icon(200, 240)
    paste_scaled(canvas, win, cx + 10, cy + 10, lerp(0.7, 1, ease_out_back((t - t0) / 0.3)), a)
    lay = Image.new("RGBA", (400, 360), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    c = YELLOW + (255,)
    pw = ease_out_cubic((t - ev["w"]) / 0.35)
    if pw > 0:   # width line above the window
        y = 38
        half = 100 * pw
        d.line((210 - half, y, 210 + half, y), fill=c, width=5)
        for x in (210 - half, 210 + half):
            d.line((x, y - 14, x, y + 14), fill=c, width=5)
        d.polygon([(210 - half, y), (210 - half + 16, y - 9), (210 - half + 16, y + 9)], fill=c)
        d.polygon([(210 + half, y), (210 + half - 16, y - 9), (210 + half - 16, y + 9)], fill=c)
    ph = ease_out_cubic((t - ev["h"]) / 0.35)
    if ph > 0:   # height line left of the window
        x = 52
        half = 120 * ph
        d.line((x, 190 - half, x, 190 + half), fill=c, width=5)
        for y in (190 - half, 190 + half):
            d.line((x - 14, y, x + 14, y), fill=c, width=5)
        d.polygon([(x, 190 - half), (x - 9, 190 - half + 16), (x + 9, 190 - half + 16)], fill=c)
        d.polygon([(x, 190 + half), (x - 9, 190 + half - 16), (x + 9, 190 + half - 16)], fill=c)
    al = lay.getchannel("A").point(lambda v: int(v * a)); lay.putalpha(al)
    canvas.alpha_composite(lay, (int(cx - 200), int(cy - 180)))

def draw_percent(canvas, t, ev):
    """Progress ring counting up to 70%."""
    t0, t1 = ev["t0"], ev["t1"]
    a = ease_out_cubic((t - t0) / 0.2) * clamp((t1 - t) / 0.2)
    cx, cy = ev["pos"]
    R = 150
    glass_panel(canvas, (cx - R - 30, cy - R - 30, cx + R + 30, cy + R + 30), radius=R + 30, a=a)
    p = ease_out_cubic((t - t0 - 0.05) / 0.7)
    val = 70 * p
    img = new_layer(2 * R, 2 * R)
    d = ImageDraw.Draw(img)
    u = SS
    th = 26 * u
    d.ellipse((th / 2, th / 2, 2 * R * u - th / 2, 2 * R * u - th / 2), outline=(255, 255, 255, 50), width=int(th))
    if val > 0.5:
        d.arc((th / 2, th / 2, 2 * R * u - th / 2, 2 * R * u - th / 2), -90, -90 + 360 * val / 100,
              fill=YELLOW + (255,), width=int(th))
    f = font(FONT_BLACK, 110 * u)
    s = f"{int(round(val))}%"
    b = d.textbbox((0, 0), s, font=f)
    d.text((R * u - (b[0] + b[2]) / 2, R * u - (b[1] + b[3]) / 2), s, font=f, fill=WHITE)
    img = finish(img)
    sc = ease_out_back((t - t0) / 0.35) if t - t0 < 0.35 else 1 + 0.03 * math.sin((t - t0) * 4)
    paste_scaled(canvas, img, cx, cy, sc, a)

def draw_arrow(canvas, t, ev):
    """Curved arrow pointing at the window the speaker shows."""
    t0, t1 = ev["t0"], ev["t1"]
    a = clamp((t - t0) / 0.1) * clamp((t1 - t) / 0.2)
    p = ease_out_cubic((t - t0) / 0.35)
    cx, cy = ev["pos"]
    img = new_layer(300, 260)
    d = ImageDraw.Draw(img)
    u = SS
    pts = []
    n = int(30 * p) + 2
    for i in range(n):
        q = i / 29
        # quadratic bezier from (260,220) via (240,60) to (50,60)
        x = (1 - q) ** 2 * 260 + 2 * (1 - q) * q * 230 + q * q * 60
        y = (1 - q) ** 2 * 220 + 2 * (1 - q) * q * 50 + q * q * 70
        pts.append((x * u, y * u))
    d.line(pts, fill=YELLOW + (255,), width=16 * u, joint="curve")
    if p > 0.95:
        x, y = pts[-1]
        d.polygon([(x - 34 * u, y), (x + 8 * u, y - 30 * u), (x + 8 * u, y + 30 * u)], fill=YELLOW + (255,))
    img = with_shadow(finish(img), 10, 5, 140)
    bob = math.sin((t - t0) * 9) * 6
    paste_scaled(canvas, img, cx + bob, cy, 1, a)

def draw_badge(canvas, t, ev):
    t0, t1 = ev["t0"], ev["t1"]
    spr = ev["_spr"]
    dt = t - t0
    s = ease_out_back(dt / 0.32) if dt < 0.32 else 1.0
    out = clamp((t1 - t) / 0.18)
    s *= lerp(0.7, 1, out)
    a = clamp(dt / 0.08) * out
    bob = math.sin(dt * 4.2) * 6
    rot = 0
    if ev.get("wiggle") and dt < 0.6:
        rot = math.sin(dt * 40) * 12 * (1 - dt / 0.6)
    cx, cy = ev["pos"]
    if ev.get("rings"):   # pulse rings (phone call)
        for k in range(2):
            q = ((dt * 1.4 + k * 0.5) % 1.0)
            r = 95 + q * 80
            ring = Image.new("RGBA", (int(2 * r + 8), int(2 * r + 8)), (0, 0, 0, 0))
            ImageDraw.Draw(ring).ellipse((4, 4, 2 * r + 4, 2 * r + 4), outline=GREEN + (int(200 * (1 - q) * a),), width=6)
            canvas.alpha_composite(ring, (int(cx - r - 4), int(cy + bob - r - 4)))
    paste_scaled(canvas, spr, cx, cy + bob, s, a, rot=rot)
    if ev.get("tag"):     # small secondary glyph (e.g. "?" next to the coin)
        tg = ev["_tag"]
        q = (dt - 0.2) / 0.3
        if q > 0:
            paste_scaled(canvas, tg, cx + 78, cy - 70 + bob, ease_out_back(q) if q < 1 else 1, a)

L_, R_, C_ = 250, 830, 540     # x columns for side icons
UY = 560                        # icon row when captions are at the bottom
LY = 1330                       # icon row when captions are on top (close-ups)

def build_events():
    E = []
    def B(t0, t1, pos, color, glyph, arg=None, sfx="pop", **kw):
        E.append(dict(kind="badge", t0=t0, t1=t1, pos=pos, color=color, glyph=glyph, arg=arg, sfx=sfx, **kw))
    # hook (B&W flash-forward)
    B(0.10, 1.10, (R_, UY), GREEN, "phone", rings=True, sfx="ring")
    B(2.80, 3.60, (L_, UY), RED, "trash", wiggle=True, sfx="pop")
    B(3.40, 4.05, (R_, UY), ORANGE, "warn", sfx="error", wiggle=True)
    # "все окна стандартные, кроме 2 крайних ... отличаются по размеру"
    E.append(dict(kind="windows", mode="standard", t0=5.40, t1=9.95, pos=(C_, 520), hl=6.80, diff=8.90,
                  sfx_list=[(5.40, "pop"), (6.80, "ding"), (8.90, "swish")]))
    # "зашита гипсокартоном"
    E.append(dict(kind="drywall", mode="cover", t0=11.90, t1=13.25, slide=12.25, pos=(C_, 520),
                  sfx_list=[(11.90, "pop"), (12.25, "slide")]))
    B(14.40, 15.20, (R_, UY), BLUE, "magnifier", sfx="pop")
    E.append(dict(kind="crane", t0=15.55, t1=16.60, pos=(C_, 520), sfx_list=[(15.55, "whoosh")]))
    E.append(dict(kind="tape", t0=17.05, t1=18.10, pos=(C_, 540), sfx_list=[(17.05, "pop"), (17.20, "zip")]))
    E.append(dict(kind="windows", mode="measured", t0=18.15, t1=19.95, pos=(C_, 520), m1=18.70, m2=19.40,
                  sfx_list=[(18.15, "pop"), (18.70, "ding"), (19.40, "ding")]))
    B(21.90, 22.40, (R_, UY), GREEN, "doc", arg=True, sfx="ding")
    B(23.95, 25.60, (L_, UY), ORANGE, "hammer", wiggle=True, sfx="pop")
    E.append(dict(kind="drywall", mode="reveal", t0=26.20, t1=27.65, slide=26.45, pos=(C_, 520),
                  sfx_list=[(26.20, "pop"), (26.45, "slide")]))
    E.append(dict(kind="tape", t0=28.15, t1=29.55, pos=(C_, 540), dir=-1, sfx_list=[(28.15, "pop"), (28.30, "zip")]))
    E.append(dict(kind="arrow", t0=30.45, t1=31.40, pos=(330, 640), sfx_list=[(30.45, "swish")]))
    E.append(dict(kind="dims", t0=32.30, t1=33.70, w=32.67, h=33.27, pos=(C_, 520),
                  sfx_list=[(32.30, "pop"), (32.67, "tick"), (33.27, "tick")]))
    E.append(dict(kind="windows", mode="random", t0=35.80, t1=37.15, pos=(C_, 520), warn=36.90,
                  sfx_list=[(35.80, "pop")]))
    B(37.87, 38.55, (C_, LY), RED, "cross", wiggle=True, sfx="error")
    E.append(dict(kind="windows", mode="random", t0=39.35, t1=40.00, pos=(C_, LY), sfx_list=[(39.35, "pop")]))
    B(40.05, 41.05, (R_, UY), GREEN, "phone", rings=True, sfx="ring")
    B(42.55, 43.35, (L_, UY), RED, "trash", wiggle=True, sfx="pop")
    B(43.30, 43.80, (R_, UY), ORANGE, "warn", wiggle=True, sfx="error")
    E.append(dict(kind="percent", t0=45.05, t1=46.50, pos=(C_, 500), sfx_list=[(45.05, "rise"), (45.75, "ding")]))
    B(47.15, 47.70, (R_, UY), BLUE, "arrows_v", arg=True, sfx="pop")
    B(48.20, 48.80, (L_, UY), BLUE, "arrows_h", arg=False, sfx="pop")
    B(50.50, 51.60, (C_, 520), BLUE, "refresh", sfx="swish")
    B(51.62, 52.08, (C_, 520), GREEN, "doc", arg=True, sfx="ding")
    B(52.70, 53.25, (R_, UY), BLUE, "building", sfx="pop")
    B(53.98, 54.45, (R_, LY), RED, "cross", wiggle=True, sfx="error")
    B(55.28, 56.15, (L_, LY), GREEN, "shield", sfx="ding")
    B(56.98, 58.30, (R_, LY), YELLOW, "ruble", sfx="cash", tag=True)
    B(58.37, 58.86, (L_, LY), BLUE, "bubble", sfx="pop")
    return E

GLYPHS = {"phone": g_phone, "trash": g_trash, "magnifier": g_magnifier, "hammer": g_hammer,
          "check": g_check, "cross": g_cross, "building": g_building, "shield": g_shield,
          "bubble": g_bubble, "doc": g_doc, "refresh": g_refresh, "ruble": g_ruble,
          "arrows_v": g_arrows, "arrows_h": g_arrows}

def warn_sprite(size=200):
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = [(S * 0.5, S * 0.08), (S * 0.95, S * 0.88), (S * 0.05, S * 0.88)]
    d.polygon(pts, fill=ORANGE + (255,))
    d.line(pts + [pts[0]], fill=WHITE, width=int(S * 0.045), joint="curve")
    f = font(FONT_BLACK, int(S * 0.5))
    b = d.textbbox((0, 0), "!", font=f)
    d.text((S * 0.5 - (b[0] + b[2]) / 2, S * 0.6 - (b[1] + b[3]) / 2), "!", font=f, fill=WHITE)
    return with_shadow(finish(img))

def prepare_events(E):
    for ev in E:
        if ev["kind"] != "badge": continue
        if ev["glyph"] == "warn":
            ev["_spr"] = warn_sprite()
        else:
            ev["_spr"] = badge(ev["color"], GLYPHS[ev["glyph"]], 190, ev["arg"])
        if ev.get("tag"):
            ev["_tag"] = badge(WHITE, g_text, 80, ("?", 0.6, DARK + (255,)))
    return E

DRAW = {"badge": draw_badge, "windows": draw_windows_row, "drywall": draw_drywall, "tape": draw_tape,
        "crane": draw_crane, "dims": draw_dimensions, "percent": draw_percent, "arrow": draw_arrow}

# ----------------------------------------------------------------- misc overlays
def draw_rec(canvas, t):
    """Small blinking REC tag during the B&W hook (replaces the cropped viewfinder text)."""
    if t >= CUTS[0]: return
    a = clamp((CUTS[0] - t) / 0.1)
    on = (t % 1.0) < 0.6
    d = ImageDraw.Draw(canvas)
    f = font(FONT_XB, 40)
    x, y = 90, 250
    if on:
        d.ellipse((x, y + 6, x + 30, y + 36), fill=(255, 45, 45, int(255 * a)))
    d.text((x + 44, y), "REC", font=f, fill=(255, 255, 255, int(235 * a)))

def flash_amount(t):
    c = CUTS[0]
    if c - 0.04 <= t < c + 0.2:
        return 0.75 * (1 - clamp((t - c) / 0.2))
    return 0.0

@lru_cache(None)
def vignette():
    import numpy as np
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    a = (np.clip((r - 0.55) / 0.6, 0, 1) ** 1.6 * 85).astype(np.uint8)
    v = Image.new("RGBA", (W, H), (0, 0, 0, 0)); v.putalpha(Image.fromarray(a, "L"))
    return v

# ----------------------------------------------------------------- frame assembly
def overlay_frame(t, chunks, events, base=None):
    canvas = base if base is not None else Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for ev in events:
        if ev["t0"] <= t < ev["t1"]:
            DRAW[ev["kind"]](canvas, t, ev)
    draw_rec(canvas, t)
    draw_captions(canvas, t, chunks)
    return canvas

def render(mode, src, out, f0, f1):
    chunks = build_chunks()
    events = prepare_events(build_events())
    grade = ("fps=30,"
             "eq=contrast=1.06:saturation=1.07:gamma=0.98,"
             "curves=all='0/0 0.25/0.235 0.5/0.5 0.75/0.77 1/1',"
             "colorbalance=rs=-0.02:bs=0.03:rh=0.02:bh=-0.02")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{f0 / FPS:.4f}", "-i", src, "-t", f"{(f1 - f0) / FPS + 0.1:.4f}",
                            "-vf", grade, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    if mode == "overlay":
        enc_args = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]
        in_fmt = "rgba"
    elif out.endswith(".png"):
        enc_args, in_fmt = ["-frames:v", "1", "-update", "1"], "rgb24"
    else:
        enc_args = ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-profile:v", "high", "-movflags", "+faststart"]
        in_fmt = "rgb24"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-"] + enc_args + [out], stdin=subprocess.PIPE)
    fsz = SW * SH * 3
    vig = vignette()
    for fi in range(f0, f1):
        t = fi / FPS
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            if fi == f0: raise SystemExit("no frames decoded")
            buf = last
        last = buf
        if mode == "overlay":
            frame = overlay_frame(t, chunks, events)
            enc.stdin.write(frame.tobytes()); continue
        src_img = Image.frombuffer("RGB", (SW, SH), buf, "raw", "RGB", 0, 1)
        img = src_img.resize((W, H), Image.LANCZOS, box=camera_box(t))
        img = img.filter(ImageFilter.UnsharpMask(radius=2.2, percent=55, threshold=2))
        canvas = img.convert("RGBA")
        canvas.alpha_composite(vig)
        if mode == "full":
            overlay_frame(t, chunks, events, canvas)
        fl = flash_amount(t)
        if fl > 0:
            canvas = Image.blend(canvas, Image.new("RGBA", (W, H), (255, 255, 255, 255)), fl)
        enc.stdin.write(canvas.convert("RGB").tobytes())
    enc.stdin.close(); enc.wait(); dec.kill()

def dump_events(out):
    cues = []
    for ev in build_events():
        if "sfx_list" in ev:
            cues += [{"t": t, "sfx": s} for t, s in ev["sfx_list"]]
        elif ev.get("sfx"):
            cues.append({"t": ev["t0"], "sfx": ev["sfx"]})
    cues += [{"t": c - 0.18, "sfx": "whoosh"} for c in CUTS]
    cues.append({"t": CUTS[0] - 0.02, "sfx": "impact"})
    chunks = build_chunks()
    # soft accent on numbers / key emphasis words
    for c in chunks:
        for w, t, _, _ in c["placed"]:
            if BIG.get(w, 1) > 1: cues.append({"t": t, "sfx": "boom"})
    json.dump(sorted(cues, key=lambda c: c["t"]), open(out, "w"), indent=1)
    srt = []
    for i, c in enumerate(chunks):
        def ts(x): return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(round(x % 1 * 1000)) % 1000:03d}"
        srt.append(f"{i + 1}\n{ts(c['start'])} --> {ts(c['end'])}\n{' '.join(w for w, _ in c['words'])}\n")
    open(os.path.join(os.path.dirname(out), "captions.srt"), "w", encoding="utf-8").write("\n".join(srt))
    json.dump([{"start": c["start"], "end": c["end"], "words": c["words"]} for c in chunks],
              open(os.path.join(os.path.dirname(out), "captions_words.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "events":
        dump_events(sys.argv[2]); sys.exit()
    src, out = sys.argv[2], sys.argv[3]
    total = int(DURATION * FPS)
    f0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    f1 = int(sys.argv[5]) if len(sys.argv) > 5 else total
    render(mode, src, out, f0, f1)
