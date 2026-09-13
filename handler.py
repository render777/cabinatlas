import io
import base64
import torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

# Global pipeline variable (loaded once when worker starts)
pipe = None
wireframe_img = None
ao_img = None

def init_pipeline():
    global pipe, wireframe_img, ao_img
    if pipe is not None:
        return

    # Load ControlNet models into GPU
    canny_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    depth_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[canny_controlnet, depth_controlnet],
        torch_dtype=torch.float16
    ).to("cuda")

    # Enable memory efficiency
    pipe.enable_xformers_memory_efficient_attention()

    # Preload reference geometry passes
    wireframe_img = Image.open("subdiv1_2.jpg").convert("RGB").resize((1344, 768))
    ao_img = Image.open("CitationLongitudeAO.jpg").convert("RGB").resize((1344, 768))

def handler(job):
    """
    Job input schema:
    {
      "input": {
        "seat_material": "Bone alabaster leather...",
        "veneer_material": "Matte open-pore...",
        "sidewall_material": "Greige textured linen...",
        "carpet_material": "Taupe wool loop...",
        "plating_material": "Satin champagne gold..."
      }
    }
    """
    job_input = job.get("input", {})

    seat_material = job_input.get("seat_material", "Bone alabaster leather")
    veneer_material = job_input.get("veneer_material", "Matte open-pore smoked eucalyptus")
    sidewall_material = job_input.get("sidewall_material", "Greige textured linen")
    carpet_material = job_input.get("carpet_material", "Taupe wool loop")
    plating_material = job_input.get("plating_material", "Satin champagne gold")

    # Construct targeted interior prompt
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

    # Execute diffusion pass
    output = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=[wireframe_img, ao_img],
        controlnet_conditioning_scale=[0.85, 0.60],
        num_inference_steps=28,
        guidance_scale=7.5
    ).images[0]

    # Convert generated output to base64
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_data": f"data:image/jpeg;base64,{img_b64}"
    }

if __name__ == "__main__":
    init_pipeline()
    runpod.serverless.start({"handler": handler})