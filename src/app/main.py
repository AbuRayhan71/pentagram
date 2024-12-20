import modal
import io
from fastapi import Response, HTTPException,Query,Request
from datetime import datetime,timezone

import os



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
    image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    return Response(content=buffer.getvalue(), media_type="image/jpeg")
