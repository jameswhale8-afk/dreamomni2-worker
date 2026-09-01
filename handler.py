import runpod
import base64
import io
from PIL import Image

def handler(event):
    input_data = event.get("input", {})
    image_base64 = input_data.get("image")
    instruction = input_data.get("instruction", "change hairstyle")
    
    if not image_base64:
        return {"error": "No image provided"}
    
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        result_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        return {"output": result_base64}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
