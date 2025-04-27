import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image
import torch.distributed as dist


# Download models
#snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
# Download example image
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )
image = Image.open("data/examples/wan/lady2.png")

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
from xfuser.core.distributed import (initialize_model_parallel,
                                     init_distributed_environment)
init_distributed_environment(
    rank=dist.get_rank(), world_size=dist.get_world_size())

initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=1,
    ulysses_degree=dist.get_world_size(),
)
torch.cuda.set_device(dist.get_rank())

pipe = WanVideoPipeline.from_model_manager(model_manager,
                                           torch_dtype=torch.bfloat16,
                                           device=f"cuda:{dist.get_rank()}",
                                           use_usp=True if dist.get_world_size() > 1 else False)
pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Image-to-video
video = pipe(
    #prompt="A stunningly beautiful and charming East Asian woman in her late twenties to early thirties is hosting a talk show or delivering a presentation in a luxurious, well-lit studio. She has flawless, radiant skin, soft features, and expressive, sparkling eyes that convey warmth and intelligence. Her long, silky hair is elegantly styled, complementing her graceful appearance. Seated at a round, black glass table with golden accents, she has a sleek silver laptop open in front of her. She wears a pastel pink tailored blazer over a mint green silk blouse, perfectly blending professional elegance with a soft, approachable charm. Her friendly smile, gentle voice, and animated hand gestures make her presence both captivating and adorable. The background features a tastefully designed interior with warm ambient lighting, wooden panels, neatly arranged bookshelves, and minimalist décor, creating a cozy yet sophisticated atmosphere. Smooth camera movement captures her every expression and gesture, highlighting her beauty, charisma, and confident professionalism.",
    prompt="A candid, realistic video capturing a 21-year-old woman of European descent with fair skin, long blonde hair, and bright blue eyes. She is stunningly attractive, with youthful, delicate facial features and a warm, approachable expression. The woman is seated indoors in a cozy, well-lit room with soft, natural daylight filtering through a nearby window. She’s dressed in a simple, elegant woolen dress with a small microphone clipped to it, suggesting she’s participating in a casual interview or conversation. She gently smiles, occasionally nodding as she listens attentively, her eyes reflecting curiosity and warmth. The atmosphere feels relaxed and inviting, with subtle movements emphasizing her engagement in the discussion.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    height=1024,
    width=680,
    num_inference_steps=50,
    seed=0, tiled=True
)
if dist.get_rank() == 0:
    save_video(video, "mi300_wan14_i2v_720p_lady2_1.mp4", fps=15, quality=9)
