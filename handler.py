import io
import os
import base64
import torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

pipe = None
wireframe_img = None
ao_img = None

def get_pipeline():
    global pipe, wireframe_img, ao_img
    if pipe is not None:
        return pipe

    print("--> Loading ControlNet models...")
    canny_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    depth_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    )

    print("--> Loading SDXL Base pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[canny_controlnet, depth_controlnet],
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Use PyTorch 2.x native scaled dot-product attention (no xformers needed)
    pipe.enable_attention_slicing()

    # Pre-load reference maps
    if os.path.exists("subdiv1_2.jpg"):
        wireframe_img = Image.open("subdiv1_2.jpg").convert("RGB").resize((1344, 768))
    if os.path.exists("CitationLongitudeAO.jpg"):
        ao_img = Image.open("CitationLongitudeAO.jpg").convert("RGB").resize((1344, 768))

    print("--> Pipeline successfully initialized!")
    return pipe

def handler(job):
    pipeline = get_pipeline()
    job_input = job.get("input", {})

    seat_material = job_input.get("seat_material", "Bone alabaster leather")
    veneer_material = job_input.get("veneer_material", "Matte open-pore smoked eucalyptus")
    sidewall_material = job_input.get("sidewall_material", "Greige textured linen")
    carpet_material = job_input.get("carpet_material", "Taupe wool loop")
    plating_material = job_input.get("plating_material", "Satin champagne gold")

    positive_prompt = (
        f"Luxury private business jet interior, Cessna Citation Longitude cabin. "
        f"Executive seats upholstered in {seat_material}. "
        f"Bulkheads and tables finished in {veneer_material}. "
        f"Lower sidewalls finished in {sidewall_material}. "
        f"Center aisle carpet featuring {carpet_material}. "
        f"Metal trim, drinkrails, and latches in {plating_material}. "
        f"China white headliner, warm ambient LED cove lights, daylight through windows. "
        f"8k architectural interior photography, photorealistic, octane render."
    )

    negative_prompt = (
        "distorted windows, uneven bulkheads, blurry seams, warped airframe, "
        "deformed furniture, low resolution, noise"
    )

    output = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=[wireframe_img, ao_img],
        controlnet_conditioning_scale=[0.85, 0.60],
        num_inference_steps=28,
        guidance_scale=7.5
    ).images[0]

    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image_data": f"data:image/jpeg;base64,{img_b64}"}

runpod.serverless.start({"handler": handler})