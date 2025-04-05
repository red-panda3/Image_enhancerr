from transformers import AutoModelForImageSuperResolution, AutoFeatureExtractor
import torch
from PIL import Image
import requests

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("fal/AuraSR-v2")
model = AutoModelForImageSuperResolution.from_pretrained("fal/AuraSR-v2")

# Load an image (you can use a local image or download one)
image_url = "https://plus.unsplash.com/premium_photo-1677181729163-33e6b59d5c8f?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with a valid image URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    output = model(**inputs)

# The output is a tensor; convert it to a PIL image
output_image = output["pixel_values"].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
output_image = (output_image * 255).astype("uint8")  # Convert to uint8 format
output_image = Image.fromarray(output_image)

# Save or display the output image
output_image.save("output_image.png")
output_image.show()