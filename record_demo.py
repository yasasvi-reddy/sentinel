#!/usr/bin/env python3
"""
record_demo.py — Automated Sentinel demo recording

What this script does:
  1. Checks both servers (FastAPI :8000, Vite :5173)
  2. Opens Chrome at architecture.html, maximises window
  3. Pre-generates all narration audio (Samantha, 170 wpm)
  4. Starts ffmpeg screen capture (avfoundation, 30 fps)
  5. Automates Chrome: architecture → landing (URL-param auto-fill) →
     Kharkiv analysis → results → Mariupol
  6. Stops recording, builds precisely-timed audio track
  7. Merges video + audio, crops macOS menu bar from top
  8. Saves results/sentinel_demo.mp4 + results/sentinel_demo.srt

Prerequisites:
  brew install ffmpeg
  source venv/bin/activate
  uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 --loop asyncio
  cd war-damage-ui && npm run dev

Run:
  source venv/bin/activate && python3 record_demo.py
"""

import subprocess, time, os, sys, signal, shutil
from pathlib import Path

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
TMP     = ROOT / ".demo_tmp"
TMP.mkdir(exist_ok=True)

FFMPEG    = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
VOICE     = "Samantha"
RATE      = 165            # words per minute
SCREEN    = "3"            # avfoundation: Capture screen 0
ARCH_HTML = (ROOT / "docs" / "architecture.html").resolve()
OUTPUT    = RESULTS / "sentinel_demo.mp4"

# Chrome window crop parameters (set in setup_chrome)
CROP_W, CROP_H, CROP_X, CROP_Y = 1920, 1163, 0, 37  # defaults; overridden at runtime


# ── Scene definitions ──────────────────────────────────────────────────────────
# (pause_before_s, narration_text, action_key)
SCENES = [
    # ── Architecture walkthrough ────────────────────────────────────────────
    (0.5,
     "Welcome to Sentinel — an automated war damage detection system "
     "built at the University of Florida.",
     "show_arch"),

    (0.8,
     "Sentinel uses freely available Sentinel-2 satellite imagery from the "
     "European Space Agency — freely available at 10 metres per pixel — "
     "to detect and quantify the destruction of civilian infrastructure "
     "in active conflict zones.",
     None),

    (0.8,
     "The pipeline has five stages. "
     "Stage one: acquisition. "
     "Sentinel-2 multispectral imagery is fetched from Google Earth Engine "
     "for two time windows — a pre-war baseline and a post-conflict analysis window. "
     "Cloud masking is applied via the QA60 quality band.",
     None),

    (0.8,
     "Stage two: preprocessing. "
     "The full satellite image is sliced into overlapping 256 by 256 pixel patches, "
     "normalised to reflectance values, and stacked into six-channel tensors "
     "that pair the pre-war and post-war RGB bands side by side.",
     None),

    (0.8,
     "Stage three: inference. "
     "A U-Net with a ResNet-34 encoder runs sliding-window segmentation "
     "across the entire image, producing full-resolution probability maps. "
     "A Vision Transformer then classifies each region as "
     "undamaged, newly damaged, or pre-existing damage.",
     None),

    (0.8,
     "Stage four: the FastAPI backend vectorises the damage label map "
     "into geo-referenced polygons and caches the result to disk, "
     "so repeat queries return in under one second. "
     "Stage five: a React frontend renders everything on an interactive canvas.",
     None),

    # ── Live demo ───────────────────────────────────────────────────────────
    (1.5,
     "Let's see it live. Here is the Sentinel interface.",
     "open_app"),

    (1.5,
     "The green live-data indicator confirms the FastAPI backend "
     "is online and all models are loaded.",
     None),

    (1.0,
     "I'll select Kharkiv, Ukraine — a city that experienced "
     "intense urban warfare starting in February 2022.",
     None),

    (1.5,
     "The date range is pre-populated: March through August 2022, "
     "covering the first six months of the conflict.",
     None),

    (0.8,
     "Let me flag hospitals and schools as the infrastructure "
     "types we want to highlight.",
     "click_infra"),

    (0.8,
     "Clicking Analyze. Because Kharkiv is a pre-cached location, "
     "the full pipeline result returns almost instantly.",
     "click_analyze"),

    (4.5,
     "The loading screen steps through each pipeline stage — "
     "cloud masking, U-Net segmentation, ViT temporal classification, "
     "zone vectorisation, and report generation.",
     None),

    (1.5,
     "And here are the results. "
     "The satellite map shows post-war Kharkiv with the damage mask overlaid. "
     "Red zones mark areas newly damaged since the conflict began. "
     "Orange zones indicate pre-existing damage already present before the war.",
     None),

    (0.8,
     "The metrics panel shows 2,266 damage zones flagged across the city, "
     "with over 69,000 newly damaged pixels and more than 1.1 million "
     "pixels of pre-existing damage.",
     None),

    (0.8,
     "You can click any damage zone to open a detail popup showing "
     "its classification, confidence score, and geo-coordinates.",
     "click_zone"),

    (1.5,
     "The confidence threshold slider on the left filters zones in real time. "
     "Raising it removes lower-confidence predictions and focuses the map "
     "on the areas the model is most certain about.",
     "adjust_slider"),

    (1.2,
     "Now let me show you Mariupol — a port city that suffered "
     "some of the most intense urban destruction of the entire conflict.",
     "go_mariupol"),

    (5.0,
     "Mariupol shows over 23 percent of surveyed pixels with some form "
     "of damage — compared to 18 percent for Kharkiv — "
     "reflecting the near-total urban destruction the city experienced.",
     None),

    (0.8,
     "On the evaluation side, the balanced U-Net checkpoint achieves "
     "a Mean I-o-U of 0.838 on the combined test set, "
     "with 91.4 percent overall pixel accuracy. "
     "Damaged pixel recall is 96.5 percent — "
     "the model catches almost all real damage with very few misses.",
     None),

    (0.8,
     "Kharkiv scores a Mean I-o-U of 0.849. "
     "Mariupol is lower at 0.673, "
     "reflecting the smaller amount of Mariupol training data — "
     "an area for continued improvement as more imagery becomes available.",
     None),

    (0.8,
     "Sentinel is open source, built on PyTorch, FastAPI, and React, "
     "and designed to scale to any conflict zone where "
     "Sentinel-2 imagery is available. Thank you for watching.",
     None),
]


# ── Browser automation ─────────────────────────────────────────────────────────
def run_js(js: str):
    """Execute JavaScript in Chrome's front tab via osascript."""
    safe = js.replace('"', '\\"').replace("\n", " ")
    script = (
        'tell application "Google Chrome" to '
        f'execute front window\'s active tab javascript "{safe}"'
    )
    subprocess.run(["osascript", "-e", script], capture_output=True)


def bring_chrome_front():
    subprocess.run(["osascript", "-e",
        'tell application "Google Chrome" to activate'], capture_output=True)
    time.sleep(0.3)


def chrome_navigate(url: str):
    """Navigate Chrome's front tab to a URL."""
    escaped = url.replace('"', '\\"')
    subprocess.run(["osascript", "-e",
        f'tell application "Google Chrome" to open location "{escaped}"'],
        capture_output=True)
    bring_chrome_front()


def setup_chrome():
    """Open Chrome at architecture page, maximise to fill screen, return crop params."""
    global CROP_W, CROP_H, CROP_X, CROP_Y

    # Navigate to architecture page (or open Chrome if not running)
    chrome_navigate(f"file://{ARCH_HTML}")
    time.sleep(2.0)

    # Maximise to full screen width/height
    subprocess.run(["osascript", "-e", """
        tell application "Google Chrome"
            activate
            set bounds of front window to {0, 0, 3840, 2400}
        end tell
    """], capture_output=True)
    time.sleep(0.8)

    # Read back actual window bounds to compute crop
    result = subprocess.run(
        ["osascript", "-e",
         "tell application \"Google Chrome\" to get bounds of front window"],
        capture_output=True, text=True
    )
    try:
        b = [int(x.strip()) for x in result.stdout.strip().split(",")]
        # b = [left, top, right, bottom]
        CROP_X = max(b[0], 0)
        CROP_Y = max(b[1], 0)  # top = menu bar height
        raw_w  = b[2] - b[0]
        raw_h  = b[3] - b[1]
        # Ensure even dimensions for H.264
        CROP_W = raw_w - (raw_w % 2)
        CROP_H = raw_h - (raw_h % 2)
        print(f"  Chrome bounds: {b}  →  crop {CROP_W}×{CROP_H} at ({CROP_X},{CROP_Y})")
    except Exception as e:
        print(f"  Warning: could not parse Chrome bounds ({e}), using defaults")

    bring_chrome_front()


ACTIONS = {}

def action(name):
    def decorator(fn):
        ACTIONS[name] = fn
        return fn
    return decorator


@action("show_arch")
def show_arch():
    """Architecture page is already showing — just ensure Chrome is front."""
    bring_chrome_front()
    time.sleep(0.5)


@action("open_app")
def open_app():
    """Navigate to the app with Kharkiv pre-selected via URL param."""
    chrome_navigate("http://localhost:5173?demo=kharkiv")
    time.sleep(2.5)   # wait for React render + useEffect to run


@action("click_infra")
def click_infra():
    run_js("""
        document.querySelectorAll('button').forEach(function(b) {
            var t = b.textContent.trim();
            if (t === 'HOSPITALS' || t === 'SCHOOLS') b.click();
        });
    """)
    time.sleep(0.6)


@action("click_analyze")
def click_analyze():
    run_js("""
        document.querySelectorAll('button').forEach(function(b) {
            if (b.textContent.trim() === 'ANALYZE' && !b.disabled) b.click();
        });
    """)
    time.sleep(5.0)   # wait for loading screen to appear and start animating


@action("click_zone")
def click_zone():
    run_js("""
        var canvas = document.querySelector('canvas');
        if (canvas) {
            var rect = canvas.getBoundingClientRect();
            canvas.dispatchEvent(new MouseEvent('click', {
                clientX: rect.left + rect.width  * 0.55,
                clientY: rect.top  + rect.height * 0.40,
                bubbles: true
            }));
        }
    """)
    time.sleep(0.8)


@action("adjust_slider")
def adjust_slider():
    # Dismiss any open popup first
    run_js("""
        document.querySelectorAll('button').forEach(function(b){
            if (b.textContent.includes('×') || b.textContent.includes('Close')) b.click();
        });
    """)
    time.sleep(0.4)
    # Move confidence slider to ~85%
    run_js("""
        var s = document.querySelector('input[type=range]');
        if (s) {
            Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')
                  .set.call(s, '0.85');
            s.dispatchEvent(new Event('input', {bubbles: true}));
        }
    """)
    time.sleep(0.5)


@action("go_mariupol")
def go_mariupol():
    # Click "NEW ANALYSIS" to return to landing
    run_js("""
        document.querySelectorAll('button').forEach(function(b){
            if (b.textContent.includes('NEW ANALYSIS')) b.click();
        });
    """)
    time.sleep(2.0)
    # Navigate with Mariupol pre-selected via URL param
    chrome_navigate("http://localhost:5173?demo=mariupol")
    time.sleep(2.5)
    # Click Analyze
    run_js("""
        document.querySelectorAll('button').forEach(function(b){
            if (b.textContent.trim() === 'ANALYZE' && !b.disabled) b.click();
        });
    """)
    time.sleep(5.0)   # wait for loading → results


# ── Audio helpers ──────────────────────────────────────────────────────────────
def generate_audio(text: str, path: Path):
    """Generate narration audio, return (wav_path, duration_seconds)."""
    subprocess.run(
        ["say", "-v", VOICE, "-r", str(RATE), "-o", str(path), text],
        check=True, capture_output=True
    )
    wav_path = path.with_suffix(".wav")
    subprocess.run([FFMPEG, "-y", "-i", str(path), str(wav_path)],
                   capture_output=True)
    result = subprocess.run(
        [FFMPEG, "-i", str(wav_path), "-f", "null", "-"],
        capture_output=True, text=True
    )
    duration = 0.0
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            break
    return wav_path, duration


def seconds_to_srt_time(s: float) -> str:
    ms  = int((s % 1) * 1000)
    s   = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def wrap_caption(text: str, width: int = 60) -> str:
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width and line:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


# ── Pre-flight checks ──────────────────────────────────────────────────────────
def check_servers():
    import urllib.request, urllib.error
    for url, name in [
        ("http://localhost:8000/health", "FastAPI"),
        ("http://localhost:5173",        "Vite"),
    ]:
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"  ✓ {name} online")
        except Exception:
            print(f"  ✗ {name} not reachable — start it first, then re-run.")
            sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Sentinel Demo Recorder ===\n")

    # 1. Check servers
    print("[1/8] Checking servers …")
    check_servers()

    # 2. Open Chrome, maximise, get crop params
    print("\n[2/8] Setting up Chrome …")
    setup_chrome()

    # 3. Generate narration audio
    print(f"\n[3/8] Generating {len(SCENES)} narration segments (voice: {VOICE}) …")
    audio_segments = []
    for i, (pause, text, _) in enumerate(SCENES):
        aiff = TMP / f"seg_{i:02d}.aiff"
        wav, dur = generate_audio(text, aiff)
        audio_segments.append((pause, wav, dur, text))
        label = f"{text[:58]}…" if len(text) > 58 else text
        print(f"  [{i+1:2d}/{len(SCENES)}] {dur:.1f}s — {label}")

    total_audio = sum(p + d for p, _, d, _ in audio_segments)
    print(f"\n  Total duration: {total_audio:.1f}s ({total_audio/60:.1f} min)")

    # 4. Start screen recording
    print("\n[4/8] Starting screen recording …")
    raw_video = TMP / "raw_screen.mp4"
    rec = subprocess.Popen(
        [
            FFMPEG, "-y",
            "-f", "avfoundation",
            "-framerate", "30",
            "-capture_cursor", "1",
            "-i", SCREEN,
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(raw_video),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    print("  Recording started.\n")

    # 5. Execute scenes
    t_start = time.time()
    narration_offsets = []

    for i, ((pause, text, action_key), (_, wav, dur, _)) in \
            enumerate(zip(SCENES, audio_segments)):
        if pause > 0:
            time.sleep(pause)

        if action_key and action_key in ACTIONS:
            ACTIONS[action_key]()

        offset = time.time() - t_start
        narration_offsets.append((offset, dur))

        label = f"{text[:52]}…" if len(text) > 52 else text
        print(f"  [{i+1:2d}/{len(SCENES)}] t={offset:.1f}s  {label}")
        subprocess.run(["afplay", str(wav)])

    time.sleep(2.0)
    total_recorded = time.time() - t_start
    print(f"\n  Demo complete. Total: {total_recorded:.1f}s")

    # 6. Stop recording
    print("\n[5/8] Stopping screen recording …")
    rec.send_signal(signal.SIGINT)
    rec.wait(timeout=10)
    time.sleep(1.0)

    # 7. Build audio track
    print("\n[6/8] Building audio track …")
    combined_audio = TMP / "narration.wav"
    inputs, filter_parts, mix_inputs = [], [], ""
    for i, (wav_path) in enumerate(seg[1] for seg in audio_segments):
        inputs += ["-i", str(wav_path)]
        delay_ms = int(narration_offsets[i][0] * 1000)
        filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[a{i}]")
        mix_inputs += f"[a{i}]"
    n = len(audio_segments)
    fc = "; ".join(filter_parts) + \
         f"; {mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0[out]"
    subprocess.run(
        [FFMPEG, "-y"] + inputs + [
            "-filter_complex", fc,
            "-map", "[out]",
            "-ar", "44100",
            str(combined_audio),
        ],
        capture_output=True, check=True,
    )
    print(f"  Audio track: {combined_audio.name}")

    # 8. Generate SRT
    print("\n[7/8] Generating SRT captions …")
    srt_path = TMP / "captions.srt"
    srt_lines = []
    for i, (offset, dur) in enumerate(narration_offsets):
        start = seconds_to_srt_time(offset)
        end   = seconds_to_srt_time(offset + dur)
        srt_lines.append(
            f"{i+1}\n{start} --> {end}\n{wrap_caption(SCENES[i][1])}\n"
        )
    srt_path.write_text("\n".join(srt_lines))
    print(f"  {len(narration_offsets)} entries written")

    # 9. Merge video + audio, crop menu bar
    print("\n[8/8] Merging video + audio (cropping menu bar) …")
    RESULTS.mkdir(parents=True, exist_ok=True)
    crop_filter = f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y}"
    subprocess.run([
        FFMPEG, "-y",
        "-i", str(raw_video),
        "-i", str(combined_audio),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT),
    ], check=True)

    shutil.copy(srt_path, OUTPUT.with_suffix(".srt"))
    shutil.copy(srt_path, RESULTS / "sentinel_demo.srt")

    print(f"\n✓ Done!")
    print(f"  Video   : {OUTPUT}")
    print(f"  Captions: {RESULTS / 'sentinel_demo.srt'}")
    print(f"  Duration: {total_recorded:.0f}s")

    print("\n  Cleaning up …")
    shutil.rmtree(TMP)
    print("  Done.\n")


if __name__ == "__main__":
    main()
