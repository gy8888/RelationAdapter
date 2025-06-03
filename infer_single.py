import os
import glob
import numpy as np
from PIL import Image
import json
from tqdm import tqdm  # for displaying progress bar

import torch
import torch.nn as nn

# from pipeline_flux_ipa import FluxPipeline
from src.pipeline_pe_clone import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel

# ======================== Image Resize Function ===========================
def resize_img(input_image, pad_to_regular=False, target_long_side=512, mode=Image.BILINEAR):
    w, h = input_image.size
    aspect_ratios = [(3, 4), (4, 3), (1, 1), (16, 9), (9, 16)]

    if pad_to_regular:
        img_ratio = w / h

        # Find the aspect ratio closest to the original image
        best_ratio = min(
            aspect_ratios,
            key=lambda r: abs((r[0] / r[1]) - img_ratio)
        )

        target_w_ratio, target_h_ratio = best_ratio
        if w / h >= target_w_ratio / target_h_ratio:
            target_w = w
            target_h = int(w * target_h_ratio / target_w_ratio)
        else:
            target_h = h
            target_w = int(h * target_w_ratio / target_h_ratio)

        # Create white background and paste the image centered
        padded_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset_x = (target_w - w) // 2
        offset_y = (target_h - h) // 2
        padded_img.paste(input_image, (offset_x, offset_y))
        input_image = padded_img
        w, h = input_image.size

    # Resize while keeping aspect ratio
    scale_ratio = target_long_side / max(w, h)
    new_w = round(w * scale_ratio)
    new_h = round(h * scale_ratio)
    input_image = input_image.resize((new_w, new_h), mode)

    return input_image

# ======================== MLP Projection Module ===========================
class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

# ======================== IPAdapter Wrapper ===========================
class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # Load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        
        # Initialize image projection model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)
        
        return image_proj_model
    
    def set_ip_adapter(self):
        transformer = self.pipe.transformer
        ip_attn_procs = {}  # total 57 layers: 19 + 38
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = transformer.attn_processors[name]
    
        transformer.set_attn_processor(ip_attn_procs)
    
    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")

        # ---- 1. Load weights for image_proj_model ----
        image_proj_state_dict = state_dict["image_proj"]
        if list(image_proj_state_dict.keys())[0].startswith("module."):
            # Remove DataParallel/DDP prefix
            image_proj_state_dict = {
                k.replace("module.", ""): v for k, v in image_proj_state_dict.items()
            }
        self.image_proj_model.load_state_dict(image_proj_state_dict, strict=True)

        # ---- 2. Load weights for ip_adapter (attn_processors) ----
        ip_adapter_state_dict = state_dict["ip_adapter"]
        if list(ip_adapter_state_dict.keys())[0].startswith("module."):
            ip_adapter_state_dict = {
                k.replace("module.", ""): v for k, v in ip_adapter_state_dict.items()
            }

        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(ip_adapter_state_dict, strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, IPAFluxAttnProcessor2_0):
                attn_processor.scale = scale

    def generate(
        self,
        condition_image=None,
        pil_image=None,                  # supports list or tuple of two PIL images
        clip_image_embeds=None,
        prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=3.5,
        num_inference_steps=24,
        **kwargs,
    ):
        self.set_scale(scale)

        # Support case with two input images
        if isinstance(pil_image, (list, tuple)) and len(pil_image) == 2:
            image_prompt_embeds1 = self.get_image_embeds(pil_image=pil_image[0])
            image_prompt_embeds2 = self.get_image_embeds(pil_image=pil_image[1])
            image_prompt_embeds = torch.cat([image_prompt_embeds1, image_prompt_embeds2], dim=0)
        else:
            image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds
            )

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.pipe(
            prompt=prompt,
            condition_image=condition_image,
            image_emb=image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images

# ======================== Parameter Setup ===========================
BASE_MODEL_PATH = '../models/FLUX.1-dev'
IMAGE_ENCODER_PATH = '../models/siglip-so400m-patch14-384'
IPADAPTER_PATH = '../models/ip_adapter-100000.bin'
LORA_WEIGHTS_PATH = "../models/checkpoint-100000-lora"
LORA_WEIGHTS_FILE = "pytorch_lora_weights.safetensors"
DEVICE = "cuda:0"

# ========== Sample Image Paths ==========
# cond1_path = "assets/close-eye-cond1.jpg"
# cond2_path = "assets/close-eye-cond2.jpg"
# source_path = "assets/close-eye-src1.jpg"
# prompt = "Apply a closed-eyes expression to the person in the image."

cond1_path = "assets/close-on-cond1.jpg"
cond2_path = "assets/close-on-cond2.jpg"
source_path = "assets/close-on-src1.jpg"
prompt = "Add a model wearing the black pants along with a taupe polo shirt and white sneakers, standing in a neutral pose."

# ======================== Model Setup ===========================
transformer = FluxTransformer2DModel.from_pretrained(
    BASE_MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = FluxPipeline.from_pretrained(
    BASE_MODEL_PATH, transformer=transformer, torch_dtype=torch.bfloat16
)

pipe.load_lora_weights(LORA_WEIGHTS_PATH, weight_name=LORA_WEIGHTS_FILE)
pipe.fuse_lora()
pipe.unload_lora_weights()
print("LoRA weights loaded ✔️")

ip_model = IPAdapter(pipe, IMAGE_ENCODER_PATH, IPADAPTER_PATH, device=DEVICE, num_tokens=128)
print("IP-Adapter initialized ✔️")

# ======================== Image Preparation ===========================
image1 = Image.open(cond1_path).convert("RGB")
image2 = Image.open(cond2_path).convert("RGB")
image1 = resize_img(image1, pad_to_regular=True, target_long_side=512)
image2 = resize_img(image2, pad_to_regular=True, target_long_side=512)

condition_image = Image.open(source_path).convert("RGB")
width, height = condition_image.size

# ======================== Inference ===========================
generated_images = ip_model.generate(
    prompt=prompt,
    condition_image=condition_image,
    height=height,
    width=width,
    pil_image=[image1, image2],
    scale=1.0,
    seed=1000,
)

# ======================== Show Result ===========================
generated_images[0].save("generated.jpg")
