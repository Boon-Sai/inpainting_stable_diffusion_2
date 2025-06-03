Here's the full content you can **directly paste into your `README.md`** file (without code blocks):

---

# Object Removal Inpainting Tool using Stable Diffusion 2

This project allows you to **remove unwanted objects** from images and **realistically inpaint** the background using **Stable Diffusion 2 Inpainting** model.

## Features

* Seamless object removal from images
* Realistic background generation using custom prompts
* Batch processing support for multiple images
* Easy-to-use API with clear examples

---

## Setup Instructions

### 1. Clone the Repository

Clone this repository and navigate into the project folder.

### 2. Create a Virtual Environment

Create and activate a virtual environment using `venv`.

### 3. Install Dependencies

Install required packages using the provided `requirements.txt`.

**Note**: Required libraries include `torch`, `diffusers`, `transformers`, `Pillow`, and `numpy`.

---

## Hugging Face Authentication

Before using the model, you must authenticate with your Hugging Face account.

Run the following command to log in:

huggingface-cli login

Follow the instructions and paste your Hugging Face access token when prompted.

---

## Input Requirements

* **Original Image**: The image you want to inpaint
* **Mask Image**: A binary mask where the area to be removed is white and the rest is black

---

## How It Works

This tool uses the `stabilityai/stable-diffusion-2-inpainting` model via the Hugging Face `diffusers` library.

It performs inpainting using prompts like:

* "natural seamless fill" (default)
* Or any **custom prompt** you define (e.g., "a scenic beach with sunset")

---

## Usage Overview

### Basic Object Removal

Remove objects from a single image using a default prompt and save the result.

### Custom Prompt Support

Use your own descriptive prompt to control the style of the generated background.

### Batch Processing

Process multiple image-mask pairs and save all results into a specified directory.

---

## Directory Structure

.
├── inpainting.py             → Main inpainting script
├── README.md                 → Project documentation
├── requirements.txt          → Python dependencies
└── images/                   → Example input images and masks

---

## Troubleshooting

* Ensure your mask and original image are the same resolution
* If using GPU and face memory issues, try enabling CPU offloading
* For best results, use high-resolution images and clear binary masks

---

## License

This project is licensed under the MIT License.

---

## Credits

* Hugging Face Diffusers: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
* Stability AI - Stable Diffusion 2: [https://huggingface.co/stabilityai/stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

---
