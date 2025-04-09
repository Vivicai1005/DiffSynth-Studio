import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download


# Download models
snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="models/Wan-AI/Wan2.1-T2V-14B")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
        ],
        "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Text-to-video
video = pipe(
    prompt="A black slip-on sneaker viewed from the rear, featuring a knitted fabric upper, a pull tab at the heel, and a thick, slightly shiny black rubber sole. The background is plain white, with a clean studio lighting setup and soft shadows. The shoe design is modern and minimalistic, emphasizing comfort and casual style.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    height=842,
    width=474,
    seed=0, tiled=True
)
save_video(video, "video1.mp4", fps=25, quality=5)
