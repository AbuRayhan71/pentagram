import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone # Ensure requests is included
import os

# Define Modal Image
def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests")
    .run_function(download_model)
)

# Define Modal App
app = modal.App("sd-demo", image=image)

@app.cls(
    image=image,
    gpu="A10G"
)
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe.to("cuda")

    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):
        """Endpoint to generate an image based on the prompt."""
        try:
            image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return Response(content=buffer.getvalue(), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint()
    def health(self):
        """Lightweight endpoint to check health status."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Warm-keeping Function
@app.function(schedule=modal.Cron("*/5 * * * *"))
def keep_warm():
    """Keep the app warm by calling health and generate endpoints."""
    # Updated URLs
    health_url = "https://aburayhan71-sd-demo-model-health.modal.run"
    generate_url = "https://aburayhan71-sd-demo-model-generate.modal.run?prompt=Test%20Prompt"

    # Check health endpoint
    try:
        health_response = requests.get(health_url)
        print(f"Health check at: {health_response.json().get('timestamp', 'No timestamp found')}")
    except Exception as e:
        print(f"Failed to check health: {e}")

    # Test generate endpoint
    try:
        generate_response = requests.get(generate_url)
        print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")
    except Exception as e:
        print(f"Failed to test generate: {e}")
