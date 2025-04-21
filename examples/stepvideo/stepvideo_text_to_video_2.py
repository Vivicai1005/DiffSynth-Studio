from modelscope import snapshot_download
from diffsynth import ModelManager, StepVideoPipeline, save_video
import torch


# Download models
#snapshot_download(model_id="stepfun-ai/stepvideo-t2v", cache_dir="models")

# Load the compiled attention for the LLM text encoder.
# If you encounter errors here. Please select other compiled file that matches your environment or delete this line.
# torch.ops.load_library("/scratch1/models/stepvideo-t2v/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so")

# Load models
model_manager = ModelManager()
model_manager.load_models(
    ["/home/weicai12/models/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
    torch_dtype=torch.float32, device="cpu"
)
model_manager.load_models(
    [
        "/home/weicai12/models/stepvideo-t2v/step_llm",
        "/home/weicai12/models/stepvideo-t2v/vae/vae_v2.safetensors",
        [
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
            "/home/weicai12/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
        ]
    ],
    torch_dtype=torch.bfloat16, device="cpu"
)
pipe = StepVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

# Enable VRAM management
# This model requires 80G VRAM.
# In order to reduce VRAM required, please set `num_persistent_param_in_dit` to a small number.
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Run!
video = pipe(
    prompt="A stunningly beautiful and charming East Asian woman in her late twenties to early thirties is hosting a talk show or delivering a presentation in a luxurious, well-lit studio. She has flawless, radiant skin, soft features, and expressive, sparkling eyes that convey warmth and intelligence. Her long, silky hair is elegantly styled, complementing her graceful appearance. Seated at a round, black glass table with golden accents, she has a sleek silver laptop open in front of her. She wears a pastel pink tailored blazer over a mint green silk blouse, perfectly blending professional elegance with a soft, approachable charm. Her friendly smile, gentle voice, and animated hand gestures make her presence both captivating and adorable. The background features a tastefully designed interior with warm ambient lighting, wooden panels, neatly arranged bookshelves, and minimalist décor, creating a cozy yet sophisticated atmosphere. Smooth camera movement captures her every expression and gesture, highlighting her beauty, charisma, and confident professionalism.",
    negative_prompt="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
    num_inference_steps=30, cfg_scale=9, num_frames=204, seed=1,
    height=832, width=480
)
save_video(
    video, "video.mp4", fps=25, quality=5,
    ffmpeg_params=["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
)
