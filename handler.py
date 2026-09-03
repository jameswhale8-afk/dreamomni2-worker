import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
import numpy as np
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from runpod.serverless import VolumeCache
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dreamomni2-handler")

MODEL_CACHE_DIR = "/root/.cache/huggingface"
VOLUME_MOUNT = "/runpod-volume"
NAMESPACE = "dreamomni2-cache"

pipe = None
vlm_model = None
vlm_processor = None

def load_model():
    global pipe, vlm_model, vlm_processor
    logger.info("Starting DreamOmni2 model load...")
    start_time = time.time()

    with VolumeCache(
        dirs=[MODEL_CACHE_DIR],
        namespace=NAMESPACE,
        volume_path=VOLUME_MOUNT,
        best_effort=True
    ):
        logger.info("Loading Flux Kontext base model...")
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            safety_checker=None,
            requires_safety_checker=False
        )

        logger.info("Loading DreamOmni2 LoRA weights...")
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            "xiabs/DreamOmni2/edit_lora",
            torch_dtype=torch.bfloat16
        )

        logger.info("Applying memory optimizations...")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.to("cuda")

        logger.info("Loading Qwen2-VL model...")
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "xiabs/DreamOmni2/vlm-model",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        vlm_model.to("cuda")
        vlm_processor = AutoProcessor.from_pretrained("xiabs/DreamOmni2/vlm-model")

        load_time = time.time() - start_time
        logger.info(f"All models loaded successfully in {load_time:.2f}s!")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def preprocess_image(image_data):
    try:
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def understand_instruction_with_vlm(source_image, reference_image, instruction):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_image},
                {"type": "image", "image": reference_image},
                {"type": "text", "text": f"You are an expert hairstylist AI. The FIRST image shows a person face/head. The SECOND image shows a reference hairstyle. Instruction: {instruction}. Please provide a detailed description of how to edit the FIRST image to apply the hairstyle from the SECOND image."}
            ]
        }
    ]

    inputs = vlm_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vlm_model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

    enhanced_prompt = vlm_processor.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Enhanced prompt from VLM: {enhanced_prompt[:100]}...")
    return enhanced_prompt

def handler(job):
    logger.info(f"Processing job: {job['id']}")
    try:
        job_input = job.get("input", {})
        source_img_data = job_input.get("source_image")
        reference_img_data = job_input.get("reference_image")
        instruction = job_input.get("instruction", "").strip()

        if not source_img_data or not reference_img_data:
            return {"error": "Missing source_image or reference_image"}
        if not instruction:
            instruction = "Change the hairstyle to match the reference image"

        source_img = preprocess_image(source_img_data)
        reference_img = preprocess_image(reference_img_data)

        if vlm_model is not None:
            enhanced_prompt = understand_instruction_with_vlm(source_img, reference_img, instruction)
        else:
            enhanced_prompt = instruction

        input_images = [source_img, reference_img]
        generation_start = time.time()

        with torch.no_grad():
            result = pipe(
                prompt=enhanced_prompt,
                image=input_images,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=1024,
                width=1024,
                strength=0.85,
                generator=torch.Generator("cuda").manual_seed(42)
            )

        generation_time = time.time() - generation_start
        output_image = result.images[0]
        img_base64 = encode_image_to_base64(output_image)

        return {
            "output_image": img_base64,
            "enhanced_prompt": enhanced_prompt,
            "generation_time": generation_time,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error processing job: {e}", exc_info=True)
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    load_model()
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({
        "handler": handler,
        "max_concurrency": 1,
        "timeout": 120,
        "retry_count": 3
    })
