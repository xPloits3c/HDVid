#!/usr/bin/env python3
"""
HDVid - Video upscaler to HD / Full HD / 2K / 4K.

Two modes:
  classic  -> pure FFmpeg: denoise + Lanczos + sharpen. Fast, runs anywhere.
  ai       -> Real-ESRGAN (ncnn-vulkan): a neural network rebuilds the
              details. Runs on NVIDIA, AMD and Intel GPUs through Vulkan,
              no CUDA/PyTorch needed. The binary is downloaded automatically.

Features:
  * Auto-detected and tested GPU encoders: NVENC (NVIDIA), QSV (Intel),
    AMF (AMD), VideoToolbox (macOS). Falls back to CPU (x264/x265).
  * Pipelined AI mode: decoding, GPU upscaling and encoding run in parallel,
    frames are processed in chunks so disk usage stays at a few GB.
  * H.264 or H.265/HEVC output, optional 60 fps interpolation,
    quick test on the first N seconds, whole-folder batch processing,
    progress bar with ETA, lossless audio copy.

https://github.com/xPloits3c/HDVid  -  MIT License
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__version__ = "2.1.0"

VIDEO_EXT = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv", ".mts", ".m2ts", ".3gp"}
TARGETS = {"720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160}
MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "alac", "opus"}
HW_ORDER = ("nvenc", "qsv", "amf", "videotoolbox")

ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
              "realesrgan-ncnn-vulkan-20220424-{}.zip")
CACHE_DIR = Path.home() / ".cache" / "hdvid"
IS_WIN = platform.system() == "Windows"

# name -> (ncnn model, supported scales)
AI_MODELS = {
    "fast":    ("realesr-animevideov3", (2, 3, 4)),  # fast, great for gameplay / screen recordings
    "quality": ("realesrgan-x4plus", (4,)),          # best for real camera footage, much slower
    "anime":   ("realesrgan-x4plus-anime", (4,)),
}


# ----------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def fmt_time(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_size(b: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def in_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


class Progress:
    def __init__(self, total: float, label: str):
        self.total = max(total, 1e-6)
        self.label = label
        self.t0 = time.time()
        self.last = 0.0
        self.lock = threading.Lock()

    def update(self, cur: float, force: bool = False) -> None:
        with self.lock:
            now = time.time()
            if not force and now - self.last < 0.5:
                return
            self.last = now
            f = min(cur / self.total, 1.0)
            el = now - self.t0
            eta = el * (1 - f) / f if f > 0 else 0
            n = int(f * 30)
            sys.stdout.write(f"\r  {self.label} [{'#' * n}{'-' * (30 - n)}] {f * 100:5.1f}%  "
                             f"elapsed {fmt_time(el)}  left {fmt_time(eta)}   ")
            sys.stdout.flush()

    def done(self) -> None:
        self.update(self.total, force=True)
        print()


class StderrCapture:
    """Drains a process' stderr in a thread (avoids pipe deadlocks)."""

    def __init__(self, stream):
        self.data = b""
        self.t = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.t.start()

    def _read(self, stream):
        self.data = stream.read()

    def text(self) -> str:
        self.t.join(timeout=5)
        return self.data.decode(errors="replace")[-3000:]


def run_quiet(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


# ----------------------------------------------------------------- probing
def probe(path: Path) -> dict:
    r = run_quiet(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if r.returncode:
        raise RuntimeError(f"ffprobe: {r.stderr.strip()}")
    d = json.loads(r.stdout)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError("no video stream found")
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)

    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = abs(int(float(sd["rotation"])))
    rot = rot or abs(int(v.get("tags", {}).get("rotate", 0)))
    if rot % 180 == 90:
        w, h = h, w

    fps_str = "30/1"
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = v.get(key, "0/0")
        num, _, den = val.partition("/")
        try:
            if float(den or 1) and 0 < float(num) / float(den or 1) <= 240:
                fps_str = val
                break
        except ValueError:
            pass
    num, _, den = fps_str.partition("/")
    fps = float(num) / float(den or 1)

    dur = float(d["format"].get("duration") or v.get("duration") or 0)
    matrix = "bt601" if v.get("color_space") in ("bt470bg", "smpte170m") else "bt709"
    return {"w": w, "h": h, "fps": fps, "fps_str": fps_str, "dur": dur,
            "audio": a["codec_name"] if a else None, "matrix": matrix,
            "bit_depth": 10 if "10" in v.get("pix_fmt", "") else 8,
            "size": path.stat().st_size}


def target_size(w: int, h: int, short: int) -> tuple[int, int]:
    if w <= h:  # portrait or square
        return short, round(h * short / w / 2) * 2
    return round(w * short / h / 2) * 2, short


# ----------------------------------------------------------------- encoders
def encoder_works(name: str) -> bool:
    r = run_quiet(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                   "-i", "color=black:s=256x256:d=0.2", "-c:v", name, "-f", "null", "-"])
    return r.returncode == 0


def choose_encoder(codec: str, use_gpu: bool) -> str:
    base = "h264" if codec == "h264" else "hevc"
    if use_gpu:
        for hw in HW_ORDER:
            if encoder_works(f"{base}_{hw}"):
                return f"{base}_{hw}"
        log("[!] No working GPU encoder found: using CPU")
    return "libx264" if codec == "h264" else "libx265"


def encoder_args(enc: str, q: int, preset: str) -> list:
    if enc == "libx264":
        a = ["-c:v", enc, "-preset", preset, "-crf", str(q), "-profile:v", "high"]
    elif enc == "libx265":
        a = ["-c:v", enc, "-preset", preset, "-crf", str(q + 2), "-x265-params", "log-level=error"]
    elif enc.endswith("_nvenc"):
        a = ["-c:v", enc, "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(q + 1),
             "-b:v", "0", "-spatial_aq", "1", "-temporal_aq", "1"]
    elif enc.endswith("_qsv"):
        a = ["-c:v", enc, "-preset", "slower", "-global_quality", str(q + 2)]
    elif enc.endswith("_amf"):
        a = ["-c:v", enc, "-quality", "quality", "-rc", "cqp",
             "-qp_i", str(q), "-qp_p", str(q + 2)]
    elif enc.endswith("_videotoolbox"):
        a = ["-c:v", enc, "-q:v", str(max(1, min(100, 100 - q * 2)))]
    else:
        a = ["-c:v", enc]
    if enc.startswith(("hevc", "libx265")):
        a += ["-tag:v", "hvc1"]  # Apple / QuickTime compatibility
    return a


def color_tags(matrix: str) -> list:
    if matrix == "bt601":
        return ["-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m"]
    return ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def audio_args(info: dict) -> list:
    if not info["audio"]:
        return []
    if info["audio"] in MP4_AUDIO_OK:
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "256k"]  # e.g. PCM from .mov does not fit in MP4


def interp_filter(fps: int) -> str:
    return f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"


# ----------------------------------------------------------------- ffmpeg with progress
def run_ffmpeg_progress(cmd: list, total_s: float, label: str) -> None:
    cmd = [cmd[0], "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", *cmd[1:]]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    err = StderrCapture(p.stderr.buffer if hasattr(p.stderr, "buffer") else p.stderr)
    bar = Progress(total_s, label)
    for line in p.stdout:
        if line.startswith("out_time_us="):
            v = line.split("=", 1)[1].strip()
            if v.isdigit():
                bar.update(int(v) / 1e6)
    rc = p.wait()
    if rc:
        print()
        raise RuntimeError(err.text())
    bar.done()


# ----------------------------------------------------------------- classic mode
def upscale_classic(src: Path, dst: Path, info: dict, args, enc: str) -> None:
    nw, nh = target_size(info["w"], info["h"], TARGETS[args.target])
    vf = []
    if args.fps:
        vf.append(interp_filter(args.fps))  # before scaling: cheaper
    if args.denoise > 0:
        s = args.denoise
        vf.append(f"hqdn3d={s}:{s}:{s * 2}:{s * 2}")
    vf.append(f"scale={nw}:{nh}:flags=lanczos+accurate_rnd+full_chroma_int")
    sharpen = 0.6 if args.sharpen is None else args.sharpen
    if sharpen > 0:
        vf.append(f"unsharp=5:5:{sharpen}:3:3:0.0")
    vf.append("format=yuv420p")

    test = ["-t", str(args.test)] if args.test else []
    cmd = ["ffmpeg", "-y", *test, "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
           "-vf", ",".join(vf), *encoder_args(enc, args.crf, args.preset),
           *color_tags(info["matrix"]), *audio_args(info),
           "-movflags", "+faststart", str(dst)]
    total = min(info["dur"], args.test) if args.test else info["dur"]
    run_ffmpeg_progress(cmd, total, "classic")


# ----------------------------------------------------------------- AI mode
def get_realesrgan(custom: str | None) -> Path:
    exe_name = "realesrgan-ncnn-vulkan.exe" if IS_WIN else "realesrgan-ncnn-vulkan"
    if custom:
        return Path(custom)
    if shutil.which(exe_name):
        return Path(shutil.which(exe_name))
    d = CACHE_DIR / "realesrgan"
    found = list(d.rglob(exe_name)) if d.exists() else []
    if not found:
        plat = {"Windows": "windows", "Linux": "ubuntu", "Darwin": "macos"}.get(platform.system())
        if not plat:
            raise RuntimeError("Real-ESRGAN is not available for this OS")
        log(f"[i] Downloading Real-ESRGAN ({plat}, ~45 MB) to {d} ...")
        d.mkdir(parents=True, exist_ok=True)
        zp = d / "realesrgan.zip"
        urllib.request.urlretrieve(ESRGAN_URL.format(plat), zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)
        zp.unlink()
        found = list(d.rglob(exe_name))
        if not found:
            raise RuntimeError("Real-ESRGAN binary not found in the archive")
        if not IS_WIN:
            os.chmod(found[0], 0o755)
        if platform.system() == "Darwin":
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(d)], capture_output=True)
    return found[0]


def pick_ai_scale(model_key: str, src_short: int, target_short: int) -> tuple[str, int]:
    model, scales = AI_MODELS[model_key]
    need = target_short / src_short
    scale = next((s for s in scales if s >= need), scales[-1])
    return model, scale


def esrgan_error_hint(output: str) -> str:
    low = output.lower()
    if "llvmpipe" in low or "vkcreateinstance failed" in low or "invalid gpu device" in low:
        if in_wsl():
            return ("\n[hint] WSL cannot see your GPU through Vulkan (only the 'llvmpipe' software "
                    "device). Run HDVid from Windows (cmd/PowerShell) instead, or use --mode classic.")
        return ("\n[hint] No usable Vulkan GPU found. Update your graphics drivers "
                "(Linux: mesa-vulkan-drivers / NVIDIA driver) and check with 'vulkaninfo --summary'.")
    if "out of memory" in low or "vkallocatememory" in low:
        return "\n[hint] GPU ran out of memory: try --tile 256 or --tile 128."
    return ""


def upscale_ai(src: Path, dst: Path, info: dict, args, enc: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for AI mode:  python -m pip install pillow")

    exe = get_realesrgan(args.esrgan_bin)
    W, H = info["w"], info["h"]
    short = TARGETS[args.target]
    model, scale = pick_ai_scale(args.model, min(W, H), short)
    nw, nh = target_size(W, H, short)
    pre = ""
    if args.turbo and min(W, H) * scale > short:
        # shrink the input so the AI output lands exactly on the target:
        # fewer pixels for the GPU (e.g. 720p->1080p x2: ~2.2x faster)
        W, H = round(nw / scale), round(nh / scale)
        pre = f"scale={W}:{H}:flags=lanczos,"
        log(f"[i] Turbo: AI input reduced to {W}x{H}")
    UW, UH = W * scale, H * scale
    dur = min(info["dur"], args.test) if args.test else info["dur"]
    total_frames = max(int(round(dur * info["fps"])), 1)
    log(f"[i] AI: {model} x{scale}  ({W}x{H} -> {UW}x{UH} -> {nw}x{nh}), ~{total_frames} frames")

    work = Path(tempfile.mkdtemp(prefix="hdvid_", dir=args.tmp))
    m = info["matrix"]
    test = ["-t", str(args.test)] if args.test else []

    dec_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *test, "-i", str(src), "-map", "0:v:0",
               "-vf", f"{pre}scale=in_color_matrix={m}:flags=accurate_rnd+full_chroma_int,format=rgb24",
               "-r", info["fps_str"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    post = [f"scale={nw}:{nh}:flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix={m}:out_range=tv"]
    if args.sharpen:
        post.append(f"unsharp=5:5:{args.sharpen}:3:3:0.0")
    if args.fps:
        post.append(interp_filter(args.fps))
    post.append("format=yuv420p")
    enc_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{UW}x{UH}", "-r", info["fps_str"], "-i", "-",
               *test, "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", ",".join(post), *encoder_args(enc, args.crf, args.preset),
               *color_tags(m), *audio_args(info), "-movflags", "+faststart", str(dst)]

    esr_cmd = [str(exe), "-n", model, "-s", str(scale), "-f", "png",
               "-m", str(exe.parent / "models"), "-j", args.threads]
    if args.gpu is not None:
        esr_cmd += ["-g", str(args.gpu)]
    if args.tile:
        esr_cmd += ["-t", str(args.tile)]

    q_up: queue.Queue = queue.Queue(maxsize=1)   # chunks ready for the GPU
    q_enc: queue.Queue = queue.Queue(maxsize=1)  # chunks ready for the encoder
    errors: list = []
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
    bar = Progress(total_frames, "AI")
    frame_bytes = W * H * 3

    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dec_err = StderrCapture(dec.stderr)
    encp = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    enc_err = StderrCapture(encp.stderr)

    def save_png(buf: bytes, path: Path) -> None:
        Image.frombytes("RGB", (W, H), buf).save(path, compress_level=1)

    def load_png(path: Path) -> bytes:
        im = Image.open(path).convert("RGB")
        if im.size != (UW, UH):
            im = im.resize((UW, UH), Image.LANCZOS)
        return im.tobytes()

    def extractor() -> None:
        idx, chunk, cdir, futs = 0, 0, None, []
        try:
            while not stop.is_set():
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                if cdir is None:
                    cdir = work / f"c{chunk:05d}"
                    (cdir / "in").mkdir(parents=True)
                futs.append(pool.submit(save_png, buf, cdir / "in" / f"{idx:08d}.png"))
                idx += 1
                if len(futs) == args.chunk:
                    for f in futs:
                        f.result()
                    q_up.put(cdir)
                    cdir, futs, chunk = None, [], chunk + 1
            if cdir is not None:
                for f in futs:
                    f.result()
                q_up.put(cdir)
            if dec.wait() and not stop.is_set():
                errors.append(RuntimeError("decoder: " + dec_err.text()))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_up.put(None)

    def feeder() -> None:
        done = 0
        try:
            while True:
                cdir = q_enc.get()
                if cdir is None:
                    break
                files = sorted((cdir / "out").glob("*.png"))
                for data in pool.map(load_png, files):
                    encp.stdin.write(data)
                    done += 1
                    bar.update(done)
                shutil.rmtree(cdir, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            stop.set()
        finally:
            try:
                encp.stdin.close()
            except OSError:
                pass

    t_ext = threading.Thread(target=extractor, daemon=True)
    t_feed = threading.Thread(target=feeder, daemon=True)
    t_ext.start()
    t_feed.start()
    gpu_logged = False
    try:
        while True:
            cdir = q_up.get()
            if cdir is None or errors:
                break
            out = cdir / "out"
            out.mkdir()
            r = run_quiet([*esr_cmd, "-i", str(cdir / "in"), "-o", str(out)])
            n_in = len(list((cdir / "in").glob("*.png")))
            if r.returncode or len(list(out.glob("*.png"))) < n_in:
                msg = (r.stderr or "") + (r.stdout or "")
                raise RuntimeError("Real-ESRGAN failed:\n" + msg[-2000:] + esrgan_error_hint(msg))
            if not gpu_logged:
                gpu_logged = True
                names = sorted(set(re.findall(r"^\[(\d+) ([^\]]+)\]", r.stderr or "", re.M)))
                if names:
                    sys.stdout.write("\r" + " " * 90 + "\r")
                    log("[i] Real-ESRGAN GPU: " + ", ".join(f"{i} = {n}" for i, n in names[:1]))
            shutil.rmtree(cdir / "in", ignore_errors=True)
            q_enc.put(cdir)
    except BaseException:
        stop.set()
        for p in (dec, encp):
            p.kill()
        raise
    finally:
        q_enc.put(None)
        t_feed.join()
        t_ext.join(timeout=10)
        pool.shutdown(wait=False)
        shutil.rmtree(work, ignore_errors=True)

    rc = encp.wait()
    if errors:
        raise errors[0] if isinstance(errors[0], Exception) else RuntimeError(str(errors[0]))
    if rc:
        raise RuntimeError("encoder: " + enc_err.text())
    bar.done()


# ----------------------------------------------------------------- main
def collect_inputs(p: Path) -> list[Path]:
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and f.is_file())
    if not p.exists():
        sys.exit(f"File not found: {p}")
    return [p]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hdvid",
                                 description="HDVid - upscale videos to HD/4K (FFmpeg + Real-ESRGAN)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", type=Path, help="video file or folder")
    ap.add_argument("-o", "--output", type=Path, help="output file (single input only)")
    ap.add_argument("-O", "--outdir", type=Path, help="output folder")
    ap.add_argument("--mode", choices=("classic", "ai"), default="classic")
    ap.add_argument("--target", choices=TARGETS, default="1080p")
    ap.add_argument("--codec", choices=("h264", "hevc"), default="h264",
                    help="hevc = smaller files at the same quality")
    ap.add_argument("--crf", type=int, default=18, help="quality (lower = better, 14-23)")
    ap.add_argument("--preset", default="medium", help="x264/x265 preset when encoding on CPU")
    ap.add_argument("--cpu", action="store_true", help="force CPU encoding")
    ap.add_argument("--fps", type=int, help="interpolate to N fps (e.g. 60). SLOW")
    ap.add_argument("--denoise", type=float, default=1.5, help="[classic] denoise strength, 0 = off")
    ap.add_argument("--sharpen", type=float, help="sharpening (default 0.6 classic, 0 ai)")
    ap.add_argument("--test", type=float, help="process only the first N seconds")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    g = ap.add_argument_group("AI mode")
    g.add_argument("--model", choices=AI_MODELS, default="fast",
                   help="fast = gameplay/screen, quality = camera footage (slow), anime")
    g.add_argument("--turbo", action="store_true",
                   help="shrink the AI input so the output hits the target directly (~2x faster)")
    g.add_argument("--gpu", type=int, help="GPU id (if you have more than one)")
    g.add_argument("--tile", type=int, default=0, help="tile size (lower it if VRAM runs out, e.g. 256)")
    g.add_argument("--threads", default="2:2:2", help="Real-ESRGAN load:proc:save threads")
    g.add_argument("--chunk", type=int, default=150, help="frames per chunk (lower = less disk)")
    g.add_argument("--tmp", type=Path, help="temp folder (SSD recommended)")
    g.add_argument("--esrgan-bin", help="path to realesrgan-ncnn-vulkan, if already installed")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found in PATH. Install FFmpeg (not with pip!): see README.md")

    inputs = collect_inputs(args.input)
    if not inputs:
        sys.exit("No videos found")
    if args.output and len(inputs) > 1:
        sys.exit("-o works with a single file only; use -O for folders")

    enc = choose_encoder(args.codec, not args.cpu)
    log(f"[i] HDVid {__version__}  |  encoder: {enc}")
    if args.mode == "ai" and in_wsl():
        log("[!] WSL detected: Vulkan usually cannot see the GPU here. "
            "If AI mode fails, run HDVid from Windows.")
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    ok = 0
    for i, src in enumerate(inputs, 1):
        tag = f"{args.target}{'_ai' if args.mode == 'ai' else ''}{'_test' if args.test else ''}"
        dst = args.output or (args.outdir or src.parent) / f"{src.stem}_{tag}.mp4"
        log(f"\n=== [{i}/{len(inputs)}] {src.name}")
        if dst.exists() and not args.overwrite:
            log(f"[=] Already exists, skipping: {dst} (use --overwrite)")
            continue
        try:
            info = probe(src)
            nw, nh = target_size(info["w"], info["h"], TARGETS[args.target])
            log(f"[i] {info['w']}x{info['h']} @ {info['fps']:.2f} fps, {fmt_time(info['dur'])} -> {nw}x{nh}")
            if min(info["w"], info["h"]) >= TARGETS[args.target]:
                log("[!] Source is already at this resolution or higher")
            if info["bit_depth"] > 8:
                log("[!] 10-bit/HDR source: output will be 8-bit SDR")
            t0 = time.time()
            (upscale_ai if args.mode == "ai" else upscale_classic)(src, dst, info, args, enc)
            log(f"[+] {dst}  ({fmt_size(info['size'])} -> {fmt_size(dst.stat().st_size)}, "
                f"in {fmt_time(time.time() - t0)})")
            ok += 1
        except KeyboardInterrupt:
            dst.unlink(missing_ok=True)
            sys.exit("\n[x] Interrupted")
        except Exception as e:  # noqa: BLE001
            dst.unlink(missing_ok=True)
            log(f"\n[x] Error on {src.name}: {e}")

    log(f"\nDone: {ok}/{len(inputs)} videos in {fmt_time(time.time() - t_all)}")
    if ok < len(inputs):
        sys.exit(1)


if __name__ == "__main__":
    main()
