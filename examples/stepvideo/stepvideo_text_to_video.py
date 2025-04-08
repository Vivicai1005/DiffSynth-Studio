from modelscope import snapshot_download
from diffsynth import ModelManager, StepVideoPipeline, save_video
import torch


# Download models
#snapshot_download(model_id="stepfun-ai/stepvideo-t2v", cache_dir="models")

# Load the compiled attention for the LLM text encoder.
# If you encounter errors here. Please select other compiled file that matches your environment or delete this line.
torch.ops.load_library("/scratch1/models/stepvideo-t2v/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so")

# Load models
model_manager = ModelManager()
model_manager.load_models(
    ["/scratch1/models/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
    torch_dtype=torch.float32, device="cpu"
)
model_manager.load_models(
    [
        "/scratch1/models/stepvideo-t2v/step_llm",
        "/scratch1/models/stepvideo-t2v/vae/vae_v2.safetensors",
        [
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
            "/scratch1/models/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
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
    prompt="一位美丽的女性，年龄在25至35岁之间，身穿专业又时尚的服装，展现出她的智慧气质。她散发出一种优雅与自信的气场。她的穿着呼应了春秋季节的氛围，可能是一件时髦的西装外套搭配一件有品位的衬衫，突显出她作为主持人的身份。画面拍摄于户外的高端街区，周围可能林立着精品店或高档咖啡馆。尽管身处户外，她依然保持着专业的举止，可能正对着麦克风讲话，或与看不见的观众互动，展现出她作为主持人的从容与口才。超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。",
    negative_prompt="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
    num_inference_steps=30, cfg_scale=9, num_frames=51, seed=1
)
save_video(
    video, "video.mp4", fps=25, quality=5,
    ffmpeg_params=["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
)
