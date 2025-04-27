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
image = Image.open("data/examples/wan/dr_su_last_frame.png")

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
    #prompt="A realistic, professional video capturing a distinguished East Asian woman in her early fifties, with neatly styled short gray hair and elegant glasses. She is dressed in a dark grey tailored suit with a subtle pin on the lapel and a simple necklace, holding a few neatly organized documents in her hand. The setting is a modern corporate studio with a large screen behind her displaying part of a bold \"AMD\" logo, complemented by a textured wall and soft, neutral lighting that creates a calm and professional atmosphere. The woman has a warm, approachable smile and a gentle, engaging expression as she speaks thoughtfully about the value of education. Her bright, intelligent eyes convey wisdom and encouragement. She uses open, welcoming hand gestures, making her message feel inspiring and accessible. The camera captures her upper body in a steady frame, highlighting her poised demeanor and the sincerity in her voice, emphasizing her passion for empowering others through education.",
    prompt="The scene continues seamlessly in the same professional studio setting, with the distinguished East Asian woman seated against the backdrop of the corporate logo and softly lit textured wall. This time, her expression shifts to one of firm determination and confident authority. Her intelligent eyes now convey a strong sense of purpose and resolve as she addresses the audience directly about taking actionable steps to improve education systems. Her posture is upright and commanding, with precise, purposeful hand gestures that underscore key points. The camera subtly pushes in to highlight the intensity of her gaze and the conviction in her voice. The overall mood emphasizes leadership, clarity, and a call to action, portraying her as a visionary figure committed to driving meaningful change in education.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    height=760,
    width=1024,
    num_inference_steps=50,
    seed=0, tiled=True
)
if dist.get_rank() == 0:
    save_video(video, "mi300_wan14_i2v_720p_dr_su_1.mp4", fps=15, quality=9)
