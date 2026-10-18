"""Audio for the Reels edit: cleaned voice + synthesized SFX + quiet lo-fi bed, ducked under speech.

Usage: audio.py <src.mp4> <cues.json> <outdir>
Writes voice.wav, sfx.wav, music.wav (stems) and mix.wav (final, ~-14 LUFS).
"""
import json, subprocess, sys, wave
import numpy as np

SR = 48000
DUR = 58.86
rng = np.random.default_rng(7)

def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)

def write(path, x):
    x = np.clip(x, -1, 1)
    if x.ndim == 1: x = np.stack([x, x], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())

def tt(d): return np.arange(int(d * SR)) / SR
def env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)
def smooth(x, L):
    """Variable-length moving average (L scalar or per-sample) -> cheap lowpass."""
    L = np.maximum(1, np.broadcast_to(np.asarray(L, int), x.shape))
    c = np.concatenate([[0], np.cumsum(x)])
    i = np.arange(len(x)) + 1
    j = np.maximum(0, i - L)
    return (c[i] - c[j]) / (i - j)
def db(v): return 10 ** (v / 20)

# ------------------------------------------------------------------ sound effects
def s_pop():
    t = tt(0.09)
    f = 300 + 700 * np.exp(-t / 0.018)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.002, 0.03) * db(-11)

def s_tick():
    t = tt(0.03)
    return np.sin(2 * np.pi * 2400 * t) * env(len(t), 0.0005, 0.006) * db(-14)

def s_ding():
    t = tt(0.9)
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, a, d in
            [(1568, 1, 0.35), (3136, 0.35, 0.18), (4704, 0.15, 0.1), (2093, 0.5, 0.3)])
    return x * np.minimum(1, t / 0.002) * db(-20)

def s_ring():
    out = np.zeros(int(0.42 * SR))
    for k in range(4):
        t = tt(0.07)
        b = (np.sin(2 * np.pi * 1300 * t) + 0.6 * np.sin(2 * np.pi * 1700 * t)) * env(len(t), 0.003, 0.05)
        s = int(k * 0.1 * SR); out[s:s + len(b)] += b
    return out * db(-22)

def s_error():
    out = np.zeros(int(0.32 * SR))
    for k in range(2):
        t = tt(0.11)
        sq = np.sign(np.sin(2 * np.pi * 165 * t)) * 0.6 + np.sin(2 * np.pi * 330 * t) * 0.4
        b = smooth(sq, 6) * env(len(t), 0.004, 0.08)
        s = int(k * 0.15 * SR); out[s:s + len(b)] += b
    return out * db(-17)

def s_whoosh(d=0.38, lo=40, hi=6, vol=-19):
    n = int(d * SR)
    t = np.arange(n) / n
    noise = rng.standard_normal(n)
    L = (lo + (hi - lo) * np.sin(np.pi * t)).astype(int)      # window shrinks -> brighter mid-sweep
    x = smooth(noise, L) - smooth(noise, L * 6)
    e = np.sin(np.pi * t) ** 2
    return x / (np.abs(x).max() + 1e-9) * e * db(vol)

def s_zip():
    n = int(0.55 * SR)
    out = np.zeros(n)
    tpos, rate = 0.0, 18
    while tpos < 0.5:
        c = s_tick() * 0.7
        s = int(tpos * SR); out[s:s + len(c)] += c[:max(0, n - s)]
        tpos += 1 / rate; rate *= 1.08
    return out + s_whoosh(0.55, 30, 10, -30)[:n]

def s_rise():
    t = tt(0.65)
    f = 300 * (3 ** (t / 0.65))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.65) ** 2 * db(-25)

def s_cash():
    x = s_ding() * 1.3
    n = rng.standard_normal(int(0.12 * SR))
    burst = (n - smooth(n, 4)) * env(len(n), 0.001, 0.03) * db(-22)
    y = np.zeros(int(0.15 * SR) + len(x)); y[:len(burst)] += burst
    y[int(0.1 * SR):int(0.1 * SR) + len(x)] += x
    return y

def s_boom():
    t = tt(0.5)
    f = 40 + 70 * np.exp(-t / 0.06)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.003, 0.18) * db(-12)

def s_impact():
    n = rng.standard_normal(int(0.3 * SR))
    hiss = (n - smooth(n, 3)) * env(len(n), 0.001, 0.06) * db(-24)
    b = s_boom(); out = np.zeros(len(b)); out[:len(b)] += b; out[:len(hiss)] += hiss
    return out

SFX = {"pop": s_pop, "tick": s_tick, "ding": s_ding, "ring": s_ring, "error": s_error,
       "whoosh": lambda: s_whoosh(), "swish": lambda: s_whoosh(0.28, 25, 5, -22),
       "slide": lambda: s_whoosh(0.5, 60, 14, -21), "zip": s_zip, "rise": s_rise,
       "cash": s_cash, "boom": s_boom, "impact": s_impact}

def build_sfx(cues):
    out = np.zeros((int(DUR * SR) + SR, 2))
    for c in cues:
        x = SFX[c["sfx"]]()
        s = max(0, int(c["t"] * SR))
        pan = 0.0
        out[s:s + len(x), 0] += x * (1 - pan)
        out[s:s + len(x), 1] += x * (1 + pan)
    return out[:int(DUR * SR)]

# ------------------------------------------------------------------ lo-fi music bed
def build_music():
    bpm = 86
    beat = 60 / bpm
    # Am9 - Fmaj7 - Cmaj7 - G6 (MIDI)
    prog = [[57, 60, 64, 67, 71], [53, 57, 60, 64, 69], [48, 55, 59, 64, 67], [55, 59, 62, 64, 71]]
    roots = [45, 41, 48, 43]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    bar = 4 * beat
    nb = int(DUR / bar) + 2
    n = int((nb + 1) * bar * SR)
    L, R = np.zeros(n), np.zeros(n)
    for b in range(nb):
        ch = prog[b % 4]; t0 = b * bar
        s = int(t0 * SR)
        # warm pad
        t = tt(bar + 0.6)
        pad = sum(np.sin(2 * np.pi * hz(m) * t * (1 + det)) for m in ch for det in (-0.0015, 0.0015))
        pad *= np.minimum(1, t / 0.8) * np.minimum(1, np.maximum(0, (bar + 0.6 - t) / 0.6)) * 0.018
        e = min(n, s + len(pad)); L[s:e] += pad[:e - s]; R[s:e] += pad[:e - s]
        # soft keys: arpeggio on 8ths
        for k in range(8):
            if k in (3, 7) and b % 2: continue
            m = ch[[0, 2, 4, 3, 1, 2, 4, 3][k]] + 12
            t = tt(0.9)
            x = (np.sin(2 * np.pi * hz(m) * t) + 0.25 * np.sin(4 * np.pi * hz(m) * t)) * np.exp(-t / 0.28) * np.minimum(1, t / 0.004)
            x *= 0.05 * (0.8 + 0.2 * rng.random())
            ss = s + int((k * beat / 2 + 0.012 * rng.standard_normal()) * SR)
            e = min(n, ss + len(x)); pan = 0.3 * np.sin(k)
            L[ss:e] += x[:e - ss] * (1 - pan); R[ss:e] += x[:e - ss] * (1 + pan)
        # sub bass
        for k in (0, 2.5):
            t = tt(beat * 1.4)
            x = np.sin(2 * np.pi * hz(roots[b % 4]) * t) * np.exp(-t / 0.5) * np.minimum(1, t / 0.01) * 0.10
            ss = s + int(k * beat * SR); e = min(n, ss + len(x))
            L[ss:e] += x[:e - ss]; R[ss:e] += x[:e - ss]
        # drums: soft kick on 1 & 3, rim on 2 & 4, hats on 8ths
        for k in range(4):
            ss = s + int(k * beat * SR)
            if k in (0, 2):
                t = tt(0.3); f = 50 + 60 * np.exp(-t / 0.03)
                x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * 0.22
            else:
                nz = rng.standard_normal(int(0.12 * SR))
                x = (nz - smooth(nz, 3)) * np.exp(-np.arange(len(nz)) / SR / 0.03) * 0.04
            e = min(n, ss + len(x)); L[ss:e] += x[:e - ss]; R[ss:e] += x[:e - ss]
        for k in range(8):
            ss = s + int((k * beat / 2 + (0.03 if k % 2 else 0)) * SR)
            nz = rng.standard_normal(int(0.05 * SR))
            x = (nz - smooth(nz, 2)) * np.exp(-np.arange(len(nz)) / SR / 0.012) * (0.018 if k % 2 else 0.028)
            e = min(n, ss + len(x)); L[ss:e] += x[:e - ss] * 0.8; R[ss:e] += x[:e - ss] * 1.2
    # vinyl crackle
    cr = (rng.random(n) > 0.9993) * rng.standard_normal(n) * 0.05
    L += cr; R += cr
    m = np.stack([L, R], 1)[:int(DUR * SR)]
    # fade in/out
    f = np.ones(len(m)); fi, fo = int(1.0 * SR), int(1.5 * SR)
    f[:fi] = np.linspace(0, 1, fi); f[-fo:] = np.linspace(1, 0, fo)
    return m * f[:, None]

def main():
    src, cues_path, out = sys.argv[1:4]
    cues = json.load(open(cues_path))
    # voice: rumble cut, denoise, gentle de-mud + presence, compression, loudness
    ff("-i", src, "-vn", "-ac", "1", "-ar", str(SR), "-af",
       "highpass=f=80,afftdn=nr=12:nf=-42:tn=1,"
       "equalizer=f=250:t=q:w=1.2:g=-2,equalizer=f=3200:t=q:w=1.0:g=2.5,equalizer=f=9000:t=h:w=0.7:g=1.5,"
       "acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=2,"
       "loudnorm=I=-15:TP=-2:LRA=7", f"{out}/voice_raw.wav")
    write(f"{out}/sfx.wav", build_sfx(cues))
    write(f"{out}/music_raw.wav", build_music())
    # warm up the synth bed: lowpass + small room
    ff("-i", f"{out}/music_raw.wav", "-af", "lowpass=f=5500,aecho=0.8:0.5:60|110:0.18|0.10,volume=1.0",
       f"{out}/music.wav")
    ff("-i", f"{out}/voice_raw.wav", "-ac", "2", "-af", "atrim=0:58.86", f"{out}/voice.wav")
    # mix: music ducked by voice (sidechain), sfx on top, final loudness for Reels
    ff("-i", f"{out}/voice.wav", "-i", f"{out}/music.wav", "-i", f"{out}/sfx.wav", "-filter_complex",
       "[0:a]asplit=2[v][sc];"
       "[1:a]volume=2dB[m];"
       "[m][sc]sidechaincompress=threshold=0.05:ratio=3:attack=20:release=400:makeup=1[md];"
       "[2:a]volume=-3dB[s];"
       "[v][md][s]amix=inputs=3:normalize=0:duration=first,"
       "loudnorm=I=-14:TP=-1.5:LRA=8,aresample=48000[o]",
       "-map", "[o]", f"{out}/mix.wav")
    ff("-i", f"{out}/voice.wav", "-i", f"{out}/sfx.wav", "-filter_complex",
       "[1:a]volume=-3dB[s];[0:a][s]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=8[o]",
       "-map", "[o]", f"{out}/mix_nomusic.wav")

if __name__ == "__main__":
    main()
