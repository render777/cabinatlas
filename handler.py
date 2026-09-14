import io
import os
import sys
import base64
import cv2
import numpy as np
import torch
import runpod
from PIL import Image

pipe = None
canny_guide = None
ao_img = None

def get_pipeline():
    global pipe, canny_guide, ao_img
    if pipe is not None:
        return pipe

    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

    print("--> Loading ControlNet models...", flush=True)
    canny_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    depth_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    )

    print("--> Loading SDXL Base pipeline...", flush=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[canny_controlnet, depth_controlnet],
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Fast, photorealistic DPM++ 2M Karras scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )
    pipe.enable_attention_slicing()

    # Pre-process wireframe pass into a clean structural Canny mask
    target_res = (1344, 768)
    if os.path.exists("subdiv1_2.jpg"):
        raw_wire = cv2.imread("subdiv1_2.jpg", cv2.IMREAD_GRAYSCALE)
        raw_wire = cv2.resize(raw_wire, target_res)
        # Suppress fine mesh lines; keep only primary architectural edges
        clean_edges = cv2.Canny(raw_wire, 120, 220)
        clean_edges = clean_edges[:, :, None]
        canny_guide = Image.fromarray(np.repeat(clean_edges, 3, axis=2))

    if os.path.exists("CitationLongitudeAO.jpg"):
        ao_img = Image.open("CitationLongitudeAO.jpg").convert("RGB").resize(target_res)

    print("--> Pipeline successfully loaded and ready!", flush=True)
    return pipe

def handler(job):
    pipeline = get_pipeline()
    job_input = job.get("input", {})

    seat_material = job_input.get("seat_material", "Bone alabaster full-grain leather")
    veneer_material = job_input.get("veneer_material", "Matte open-pore smoked eucalyptus wood veneer")
    sidewall_material = job_input.get("sidewall_material", "Greige textured woven linen")
    carpet_material = job_input.get("carpet_material", "Taupe wool loop pile carpet")
    plating_material = job_input.get("plating_material", "Satin champagne brushed gold")

    positive_prompt = (
        f"Award-winning architectural interior photograph of a luxury private business jet cabin, Cessna Citation Longitude. "
        f"Executive club seats trimmed in realistic {seat_material}, subtle natural leather creases. "
        f"Bulkheads, side ledge, and fold-out tables finished in satin {veneer_material}. "
        f"Lower sidewalls upholstered in {sidewall_material}. "
        f"Aisle carpet featuring tailored {carpet_material}. "
        f"Latches, drinkrail inlay, and hardware details in {plating_material}. "
        f"Clear transparent oval windows showing bright daytime blue sky and high-altitude clouds outside. "
        f"Warm indirect LED ceiling cove lights, natural morning sun casting soft directional shadows across the cabin floor. "
        f"Shot on Hasselblad H6D-100c, 24mm architectural wide-angle lens, sharp focus, magazine quality, photorealistic, 8k."
    )

    negative_prompt = (
        "black lines, wireframe, outlines, drawing, cartoon, illustration, cel shading, "
        "sketch, solid brown windows, flat lighting, CG artifacts, oversaturated, render look, "
        "blurry, distorted geometry, lowres, dark windows"
    )

    # Dial back Canny conditioning to stop line baking while keeping structure locked
    output = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=[canny_guide, ao_img],
        controlnet_conditioning_scale=[0.38, 0.50],
        num_inference_steps=32,
        guidance_scale=8.0,
        controlnet_guidance_start=[0.0, 0.0],
        controlnet_guidance_end=[0.75, 0.85]
    ).images[0]

    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image_data": f"data:image/jpeg;base64,{img_b64}"}

runpod.serverless.start({"handler": handler})
