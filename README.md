# HDVid

**Upscale videos to HD, Full HD, 2K or 4K from the command line, with optional AI super-resolution.**

HDVid wraps FFmpeg and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) into a single Python script.
It picks the best encoder your hardware supports, runs the AI model on any Vulkan GPU (NVIDIA, AMD, Intel)
without CUDA or PyTorch, and keeps audio, colors and orientation intact.

```
python hdvid.py video.mp4 --mode ai
```

---

## Features

- **Two modes**
  - `classic`: FFmpeg only (denoise + Lanczos + sharpen). Fast, works everywhere.
  - `ai`: Real-ESRGAN rebuilds real detail instead of just enlarging pixels.
- **Any GPU for AI**: uses `realesrgan-ncnn-vulkan`, downloaded automatically on first run.
- **Hardware encoding**: auto-detects and tests NVENC (NVIDIA), Quick Sync (Intel), AMF (AMD) and VideoToolbox (macOS); falls back to x264/x265.
- **Pipelined AI processing**: decoding, GPU upscaling and encoding run in parallel; frames are processed in chunks, so disk usage stays at a few GB even for long videos.
- **H.264 or H.265/HEVC** output, optional **60 fps interpolation**.
- **Batch mode**: process a whole folder.
- **Test mode**: try your settings on the first N seconds.
- Progress bar with ETA, lossless audio copy, portrait/landscape aware, BT.709 color preserved.

## Requirements

| Component | Needed for | Install |
|---|---|---|
| Python 3.9+ | everything | [python.org](https://www.python.org/downloads/) |
| FFmpeg + FFprobe | everything | see below (**not** `pip install ffmpeg`) |
| Pillow | `--mode ai` | `python -m pip install -r requirements.txt` |
| Vulkan GPU + up-to-date drivers | `--mode ai` | NVIDIA / AMD / Intel, 4 GB VRAM recommended |

### Installing FFmpeg

```bash
# Windows (cmd or PowerShell), then reopen the terminal
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Debian / Ubuntu / Kali
sudo apt update && sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

Check it works with `ffmpeg -version`.

> ⚠️ `pip install ffmpeg` / `pip install ffmpeg-python` do **not** install FFmpeg. If you ran them, remove them with `pip uninstall ffmpeg ffmpeg-python`.

### Extra packages for AI mode on Linux

```bash
sudo apt install libvulkan1 libgomp1 vulkan-tools mesa-vulkan-drivers
vulkaninfo --summary   # your GPU must be listed
```

NVIDIA users need the proprietary driver (`sudo ubuntu-drivers autoinstall` on Ubuntu).

## Installation

```bash
git clone https://github.com/xPloits3c/HDVid.git
cd HDVid
python -m pip install -r requirements.txt
```

Optionally install it as a command:

```bash
python -m pip install .
hdvid video.mp4 --mode ai
```

## Quick start

```bash
# Full HD, classic mode
python hdvid.py video.mp4

# Try AI on the first 10 seconds first
python hdvid.py video.mp4 --mode ai --test 10

# Full video with AI
python hdvid.py video.mp4 --mode ai

# 4K with HEVC
python hdvid.py video.mp4 --mode ai --target 4k --codec hevc

# Whole folder
python hdvid.py ./videos -O ./videos_hd --mode ai
```

Output files are named `<name>_<target>[_ai][_test].mp4` next to the source (or in `-O`).

> On Windows use `python`, not `python3`: `python3` may launch the Microsoft Store Python, which does not have Pillow installed.

## Options

### General

| Option | Default | Description |
|---|---|---|
| `input` | | Video file or folder |
| `-o, --output` | | Output file (single input only) |
| `-O, --outdir` | | Output folder |
| `--mode` | `classic` | `classic` or `ai` |
| `--target` | `1080p` | `720p`, `1080p`, `1440p`, `4k` |
| `--codec` | `h264` | `h264` or `hevc` (smaller files) |
| `--crf` | `18` | Quality: lower = better (14 to 23) |
| `--preset` | `medium` | x264/x265 preset when encoding on CPU |
| `--cpu` | off | Force CPU encoding |
| `--fps N` | | Interpolate to N fps (e.g. 60). Slow |
| `--denoise` | `1.5` | Denoise strength in classic mode, `0` = off |
| `--sharpen` | `0.6` / `0` | Sharpening (classic / ai), `0` = off |
| `--test N` | | Process only the first N seconds |
| `--overwrite` | off | Overwrite existing outputs |
| `-V, --version` | | Show version |

### AI mode

| Option | Default | Description |
|---|---|---|
| `--model` | `fast` | `fast`: gameplay, screen recordings. `quality`: real camera footage (much slower). `anime`: animation |
| `--turbo` | off | Shrink the AI input so the output lands exactly on the target: about 2x faster (e.g. 720p → 1080p) |
| `--gpu N` | auto | GPU id when you have more than one (e.g. integrated + dedicated) |
| `--tile N` | `0` (auto) | Lower it (256, 128) if the GPU runs out of memory |
| `--threads` | `2:2:2` | Real-ESRGAN load:proc:save threads |
| `--chunk` | `150` | Frames per chunk: lower = less disk space used |
| `--tmp` | system temp | Temp folder (an SSD is recommended) |
| `--esrgan-bin` | auto | Path to an existing `realesrgan-ncnn-vulkan` |

## Recipes

```bash
# Gameplay / screen recording -> Full HD
python hdvid.py clip.mp4 --mode ai --model fast

# Phone footage -> Full HD, maximum quality
python hdvid.py phone.mov --mode ai --model quality --crf 16

# YouTube upload in 4K (YouTube gives 4K uploads a higher bitrate)
python hdvid.py video.mp4 --mode ai --target 4k --codec hevc

# Low-VRAM GPU (4 GB or less)
python hdvid.py video.mp4 --mode ai --tile 128

# Fast, no AI, small file
python hdvid.py video.mp4 --codec hevc --crf 20
```

## Speed

AI upscaling is heavy: every frame goes through a neural network. A 3-minute 30 fps video is about 6,500 frames.
Rough speed with `--model fast` on 720p input:

| GPU | Approx. speed |
|---|---|
| Integrated graphics (Intel UHD / Iris, AMD Vega) | 0.5 to 2 frames/s |
| Mid-range dedicated (GTX 1660, RTX 3060, RX 6600) | 5 to 15 frames/s |
| High-end (RTX 4070 and up) | 15+ frames/s |

To go faster:

1. **Use the dedicated GPU**: HDVid prints `Real-ESRGAN GPU: ...` at start. If it shows integrated graphics, try `--gpu 1`.
2. **`--turbo`**: about 2x faster when the target is not a full multiple of the source (720p → 1080p).
3. **`--model fast`** (default) is several times faster than `quality`.
4. **No dedicated GPU?** Use `--mode classic`: it processes a whole video in minutes.

## How it works

**Classic mode** runs one FFmpeg pass:
optional `minterpolate` → `hqdn3d` denoise → Lanczos `scale` → `unsharp` → encode.

**AI mode** runs a three-stage pipeline:

```
FFmpeg decode ──► PNG chunks ──► Real-ESRGAN (GPU) ──► FFmpeg scale + encode + audio
   (thread)                          (main)                    (thread)
```

The source is upscaled by the smallest model factor that reaches the target (e.g. 720p → ×2 → 1440p),
then scaled down to the exact target with Lanczos. Only one or two chunks live on disk at any time.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found in PATH` | Install FFmpeg as shown above and reopen the terminal |
| `Pillow is required for AI mode` | `python -m pip install pillow` (on Windows use `python`, not `python3`) |
| `Real-ESRGAN failed`, `llvmpipe`, `invalid gpu device` | Update GPU drivers. **On WSL the GPU is not visible to Vulkan**: run HDVid from Windows cmd/PowerShell instead |
| GPU out of memory / crash | `--tile 256` or `--tile 128` |
| Disk full | `--chunk 50` and/or `--tmp` on another drive |
| `No working GPU encoder found` | Not an error: CPU encoding is used. Update drivers or use a full FFmpeg build |
| AI is slow on a laptop | It may be using the integrated GPU: try `--gpu 1` |

Press `Ctrl+C` to stop: temp files and the partial output are removed.

## Notes

- Upscaling cannot recover information that was never recorded. AI mode makes edges and textures look much
  sharper, but heavily compressed or blurry sources have limits.
- 10-bit/HDR sources are converted to 8-bit SDR.
- Real-ESRGAN models are trained on general images; results vary with content. Use `--test` to compare models.

## Credits

- [FFmpeg](https://ffmpeg.org/)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause). The binary is downloaded from the official releases at runtime and is not redistributed in this repository.
- [ncnn](https://github.com/Tencent/ncnn) by Tencent

## License

[MIT](LICENSE) © 2026 Tuinea Doru Alin
