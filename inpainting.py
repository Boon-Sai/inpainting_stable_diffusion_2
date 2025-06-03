"""
Object Removal Inpainting Tool using Stable Diffusion 2
Removes objects and generates realistic background
"""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import os

class ObjectRemover:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        """
        Initialize the object removal inpainting model
        
        Args:
            model_id: Hugging Face model ID (default: SD2 Inpainting)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the inpainting pipeline
        print("Loading inpainting model...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Optimize for better performance
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            # Uncomment if you have VRAM issues
            # self.pipe.enable_model_cpu_offload()
        
        print("Model loaded successfully!")
    
    def remove_object(self, original_image_path, mask_image_path, output_path=None):
        """
        Remove object from image and generate background
        
        Args:
            original_image_path: Path to the original image
            mask_image_path: Path to the mask image (white = remove, black = keep)
            output_path: Path to save the result (optional)
        
        Returns:
            PIL Image with object removed
        """
        try:
            # Load images
            original_image = Image.open(original_image_path).convert("RGB")
            mask_image = Image.open(mask_image_path).convert("RGB")
            
            # Ensure images are same size
            if original_image.size != mask_image.size:
                print(f"Resizing mask to match original image size: {original_image.size}")
                mask_image = mask_image.resize(original_image.size, Image.LANCZOS)
            
            # Generate background-focused prompt for better object removal
            prompt = "natural seamless fill"
            negative_prompt = "object, person, artifact, blurry, low quality, distorted, unnatural"
            
            print("Removing object and generating background...")
            
            # Perform inpainting
            result = self.pipe(
                prompt=prompt,
                image=original_image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
            
            # Save result if output path provided
            if output_path:
                result.save(output_path)
                print(f"Result saved to: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"Error during object removal: {str(e)}")
            return None
    
    def remove_object_custom_prompt(self, original_image_path, mask_image_path, 
                                   custom_prompt="", output_path=None):
        """
        Remove object with custom background prompt
        
        Args:
            original_image_path: Path to the original image
            mask_image_path: Path to the mask image
            custom_prompt: Custom prompt for background generation
            output_path: Path to save the result
        
        Returns:
            PIL Image with object removed
        """
        try:
            # Load images
            original_image = Image.open(original_image_path).convert("RGB")
            mask_image = Image.open(mask_image_path).convert("RGB")
            
            # Ensure images are same size
            if original_image.size != mask_image.size:
                mask_image = mask_image.resize(original_image.size, Image.LANCZOS)
            
            # Use custom prompt or default
            if not custom_prompt:
                prompt = "natural background, seamless, realistic, high quality"
            else:
                prompt = custom_prompt
            
            negative_prompt = "object, person, artifact, blurry, low quality, distorted, unnatural"
            
            print(f"Removing object with prompt: '{prompt}'")
            
            # Perform inpainting
            result = self.pipe(
                prompt=prompt,
                image=original_image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
            
            # Save result if output path provided
            if output_path:
                result.save(output_path)
                print(f"Result saved to: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"Error during object removal: {str(e)}")
            return None
    
    def batch_remove_objects(self, image_mask_pairs, output_dir="results"):
        """
        Remove objects from multiple images
        
        Args:
            image_mask_pairs: List of tuples [(image_path, mask_path), ...]
            output_dir: Directory to save results
        
        Returns:
            List of result images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, (image_path, mask_path) in enumerate(image_mask_pairs):
            print(f"Processing image {i+1}/{len(image_mask_pairs)}")
            
            output_path = os.path.join(output_dir, f"result_{i+1}.jpg")
            
            result = self.remove_object(image_path, mask_path, output_path)
            
            if result:
                results.append(result)
            else:
                print(f"Failed to process: {image_path}")
        
        print(f"Batch processing complete. Results saved in: {output_dir}")
        return results

# Example usage and testing
def main():
    # Initialize the object remover
    remover = ObjectRemover()
    
    # Example 1: Simple object removal
    try:
        result = remover.remove_object(
            original_image_path="/home/litzchill/Boon_sai/inpainting/images/imgg-1.jpg",
            mask_image_path="/home/litzchill/Boon_sai/inpainting/images/masked_img.png",
            output_path="output_result.jpg"
        )
        if result:
            print("Object removal successful!")
        else:
            print("Object removal failed!")
    except FileNotFoundError:
        print("Example files not found. Please provide valid image paths.")
    
    # Example 2: Custom background prompt
    # result = remover.remove_object_custom_prompt(
    #     original_image_path="input_image.jpg",
    #     mask_image_path="mask_image.jpg",
    #     custom_prompt="beautiful garden with flowers and trees",
    #     output_path="result_custom.jpg"
    # )
    
    # Example 3: Batch processing
    # image_mask_pairs = [
    #     ("image1.jpg", "mask1.jpg"),
    #     ("image2.jpg", "mask2.jpg"),
    #     ("image3.jpg", "mask3.jpg")
    # ]
    # results = remover.batch_remove_objects(image_mask_pairs, "batch_results")

if __name__ == "__main__":
    main()