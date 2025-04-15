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
    prompt="A stunningly beautiful and charming East Asian woman in her late twenties to early thirties is hosting a talk show or delivering a presentation in a luxurious, well-lit studio. She has flawless, radiant skin, soft features, and expressive, sparkling eyes that convey warmth and intelligence. Her long, silky hair is elegantly styled, complementing her graceful appearance. Seated at a round, black glass table with golden accents, she has a sleek silver laptop open in front of her. She wears a pastel pink tailored blazer over a mint green silk blouse, perfectly blending professional elegance with a soft, approachable charm. Her friendly smile, gentle voice, and animated hand gestures make her presence both captivating and adorable. The background features a tastefully designed interior with warm ambient lighting, wooden panels, neatly arranged bookshelves, and minimalist décor, creating a cozy yet sophisticated atmosphere. Smooth camera movement captures her every expression and gesture, highlighting her beauty, charisma, and confident professionalism.",
    negative_prompt="",
    num_inference_steps=50,
    height=832,
    width=480,
    num_frames=205,
    seed=0, tiled=True
)
save_video(video, "video1.mp4", fps=25, quality=5)
