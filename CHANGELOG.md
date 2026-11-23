# Changelog

## 2.1.0 - 2026-11-23
- `--turbo`: shrinks the AI input so the output hits the target directly (~2x faster for 720p -> 1080p)
- Shows which GPU Real-ESRGAN is using
- README: speed section

## 2.0.0 - 2026-11-23
- AI mode with Real-ESRGAN (ncnn-vulkan), auto-downloaded, works on NVIDIA/AMD/Intel GPUs
- Pipelined decode → GPU upscale → encode with chunked frames (low disk usage)
- Auto-detected GPU encoders: NVENC, QSV, AMF, VideoToolbox
- HEVC output, 60 fps interpolation, batch folders, `--test` mode, progress bar with ETA
- WSL detection and clearer error hints

## 1.0.0
- First release: FFmpeg Lanczos upscaling with denoise and sharpening
