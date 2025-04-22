import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Download example image
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )
image = Image.open("data/examples/wan/lady.png")

# Image-to-video
video = pipe(
    #prompt="A stylish brown leather ankle boot is displayed in a minimalistic studio setting. The boot slowly rotates on a soft-matte white pedestal, revealing its almond-shaped toe, block heel, and smooth stitched leather texture. Subtle shadows shift across the surface under warm, diffused lighting, creating a luxurious and tactile visual experience. The background remains clean and softly blurred, allowing full focus on the elegant craftsmanship. The camera stays close, capturing the fine grain of the leather, the zipper detail at the back, and the gentle curve of the sole. ",
    prompt="A stunningly beautiful and charming East Asian woman in her late twenties to early thirties is hosting a talk show or delivering a presentation in a luxurious, well-lit studio. She has flawless, radiant skin, soft features, and expressive, sparkling eyes that convey warmth and intelligence. Her long, silky hair is elegantly styled, complementing her graceful appearance. Seated at a round, black glass table with golden accents, she has a sleek silver laptop open in front of her. She wears a pastel pink tailored blazer over a mint green silk blouse, perfectly blending professional elegance with a soft, approachable charm. Her friendly smile, gentle voice, and animated hand gestures make her presence both captivating and adorable. The background features a tastefully designed interior with warm ambient lighting, wooden panels, neatly arranged bookshelves, and minimalist décor, creating a cozy yet sophisticated atmosphere. Smooth camera movement captures her every expression and gesture, highlighting her beauty, charisma, and confident professionalism.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    height=720,
    width=480,
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "mi300_wan14_i2v_lady.mp4", fps=15, quality=9)
