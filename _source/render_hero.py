"""Render a seamless cinematic hero loop from the gazebo night photo.

Camera: slow periodic push-in toward the gazebo along the bridge with a gentle sway.
Life: horizontal ripple shimmer on the pond, twinkling of the string/solar lights,
and a few drifting warm motes. Every time-varying term is periodic in T, so the
last frame flows straight into the first.
"""
import subprocess, sys, math
import numpy as np, cv2

SRC = "/home/claude/site/src/gazebo.png"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/site/out/assets/hero-1080.mp4"
W, H = 1920, 1080
FPS = 30
SECS = 14
N = FPS * SECS
UP = 2.0

img = cv2.imread(SRC, cv2.IMREAD_COLOR)
h0, w0 = img.shape[:2]
# clean phone noise / jpeg blocks in the dark sky before upscaling
img = cv2.fastNlMeansDenoisingColored(img, None, 4, 6, 7, 21)
big = cv2.resize(img, (int(w0 * UP), int(h0 * UP)), interpolation=cv2.INTER_LANCZOS4)
blur = cv2.GaussianBlur(big, (0, 0), 1.6)
big = cv2.addWeighted(big, 1.45, blur, -0.45, 0)  # unsharp after upscale
bh, bw = big.shape[:2]
bigf = big.astype(np.float32)

# ---- masks in upscaled source space
b, g, r = [bigf[..., i] for i in range(3)]
lum = 0.114 * b + 0.587 * g + 0.299 * r
yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
water = ((b > r + 6) & (b >= g - 10) & (yy > 0.56 * bh) & (lum < 170)).astype(np.float32)
water = cv2.GaussianBlur(cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((5, 5))), (0, 0), 9)

bright = np.clip((lum - 175) / 60.0, 0, 1)
bright[: int(0.25 * bh)] = 0  # ignore sky glow
glow = cv2.GaussianBlur(bright, (0, 0), 7) * 1.0 + cv2.GaussianBlur(bright, (0, 0), 22) * 0.9
colr = cv2.GaussianBlur(bigf, (0, 0), 7) / 255.0
glow3 = (glow[..., None] * (0.55 + 0.6 * colr)).astype(np.float32)
glow3 *= np.array([0.75, 0.95, 1.25], np.float32)  # warm (BGR)

rng = np.random.default_rng(7)
phase = cv2.GaussianBlur(rng.random((bh // 16, bw // 16)).astype(np.float32), (0, 0), 1.2)
phase = cv2.resize(phase, (bw, bh), interpolation=cv2.INTER_CUBIC)
phase = (phase - phase.min()) / (phase.max() - phase.min()) * 2 * math.pi

# ---- camera
fx, fy = 815 * UP, 560 * UP          # gazebo
cx0, cy0 = bw / 2, bh / 2 + 40 * UP  # opening framing
base = max(W / bw, H / bh) * 1.02

def cam(t):
    e = (1 - math.cos(2 * math.pi * t / SECS)) / 2  # 0 -> 1 -> 0
    z = base * (1 + 0.17 * e)
    cx = cx0 + (fx - cx0) * 0.55 * e + 14 * UP * math.sin(2 * math.pi * t / SECS)
    cy = cy0 + (fy - cy0) * 0.55 * e - 4 * UP * math.sin(4 * math.pi * t / SECS)
    # affine: out = z*(src - c) + (W/2,H/2)
    M = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)
    return M, z, cx, cy

oy, ox = np.mgrid[0:H, 0:W].astype(np.float32)

# vignette + grade
vy = (oy / H - 0.5) * 2; vx = (ox / W - 0.5) * 2
vig = np.clip(1 - 0.38 * (vx ** 2 * 0.7 + vy ** 2) ** 1.2, 0.45, 1)[..., None].astype(np.float32)

# motes: periodic lissajous drifts around the gazebo (output-space-ish, source anchored)
NM = 22
mote = [dict(x=rng.uniform(560, 1100) * UP, y=rng.uniform(380, 760) * UP,
             ax=rng.uniform(10, 40) * UP, ay=rng.uniform(8, 28) * UP,
             kx=int(rng.integers(1, 3)), ky=int(rng.integers(1, 3)),
             p=rng.uniform(0, 6.28), s=rng.uniform(1.4, 3.2), kb=int(rng.integers(1, 4)))
        for _ in range(NM)]

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-maxrate", "7M", "-bufsize", "14M",
       "-pix_fmt", "yuv420p", "-profile:v", "high", "-tune", "film", "-g", str(FPS * 2),
       "-movflags", "+faststart", "-an", OUT]
ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for i in range(N):
    t = i / FPS
    M, z, cx, cy = cam(t)
    frame = cv2.warpAffine(bigf, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    wm = cv2.warpAffine(water, M, (W, H), flags=cv2.INTER_LINEAR)
    # source-space y for ripple phase
    sy = (oy - M[1, 2]) / z
    sx = (ox - M[0, 2]) / z
    w = 2 * math.pi * t / SECS
    dx = wm * (2.2 * np.sin(sy * 0.045 + 6 * w) + 1.1 * np.sin(sy * 0.11 - 9 * w + sx * 0.004))
    dy = wm * 0.6 * np.sin(sx * 0.02 + 4 * w)
    frame = cv2.remap(frame, ox + dx, oy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gl = cv2.warpAffine(glow3, M, (W, H), flags=cv2.INTER_LINEAR)
    ph = cv2.warpAffine(phase, M, (W, H), flags=cv2.INTER_LINEAR)
    # broken-up water glints (noise-gated so they read as glints, not bands)
    gate = np.clip(np.sin(ph * 4 + 2 * w) * np.sin(sx * 0.05 + ph * 3 - 3 * w), 0, 1)
    frame += (wm * 7 * gate * np.clip(np.sin(sy * 0.09 - 7 * w + sx * 0.01), 0, 1) ** 4)[..., None]

    # twinkle
    tw = 0.55 + 0.45 * np.sin(ph + 3 * w) * np.sin(ph * 1.7 + 5 * w)
    frame += gl * (tw[..., None] * 70.0)

    # motes
    ml = np.zeros((H, W), np.float32)
    for m in mote:
        mx = m["x"] + m["ax"] * math.sin(m["kx"] * w + m["p"])
        my = m["y"] + m["ay"] * math.sin(m["ky"] * w + m["p"] * 1.3)
        px, py = z * mx + M[0, 2], z * my + M[1, 2]
        br = 0.5 + 0.5 * math.sin(m["kb"] * w + m["p"] * 2)
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(ml, (int(px), int(py)), int(m["s"] * 2), float(br ** 2), -1, cv2.LINE_AA)
    ml = cv2.GaussianBlur(ml, (0, 0), 3.5)
    frame += ml[..., None] * np.array([90, 200, 255], np.float32)

    # grade: lift shadows toward ink-blue, warm highlights, vignette
    f = frame / 255.0
    f = np.clip(f, 0, 1)
    f = f * f * (3 - 2 * f) * 0.35 + f * 0.65  # gentle S curve
    f = f + np.array([0.025, 0.012, 0.0], np.float32) * (1 - f)  # BGR: blue lift in shadows
    f *= vig
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    ff.stdin.write(out.tobytes())
    if i == 0:
        cv2.imwrite("/home/claude/site/out/frame0.png", out)
    if i == N // 2:
        cv2.imwrite("/home/claude/site/out/frame_mid.png", out)
    if i % 60 == 0:
        print("frame", i, flush=True)

ff.stdin.close(); ff.wait()
print("done", OUT)
