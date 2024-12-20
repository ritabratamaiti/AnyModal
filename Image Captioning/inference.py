import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download, snapshot_download
import matplotlib.pyplot as plt

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE',
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "AnyModal/flickr30k"

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)

ds = vision.ImageDataset(dataset_name, image_processor, split = 'test')


# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)

# Initialize MultiModalModel
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1),
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="The description of the given image is: ")

if not os.path.exists("image_captioning_model"):
    os.makedirs("image_captioning_model")

# snapshot_download("AnyModal/Image-Captioning-Llama-3.2-1B", local_dir="image_captioning_model")

# Load the model
multimodal_model._load_model("image_captioning_model")

# Generate captions for a few images and plot the images and save captions in txt file


multimodal_model.eval()

os.makedirs("temp", exist_ok=True)

for _ in range(5):
    sample_idx = np.random.randint(len(ds))
    sample = ds[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"temp/image_{sample_idx}.png")

    with open(f"temp/image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")




imgs = [
    {
        'url': 'https://img.freepik.com/free-photo/people-posing-together-registration-day_23-2149096794.jpg',
        'meta': 'People'
    },
    {
        'url': 'https://discoverymood.com/wp-content/uploads/2019/01/iStock-629076332.jpg',
        'meta': 'Dog'
    }
]


# Generate captions for the daily cartoons
for idx, cartoon in enumerate(imgs):
    # download the image
    response = requests.get(cartoon['url'])
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')

    # save the image
    img.save(f"web_image_{idx}.png")

    # process the image
    image = image_processor(img, return_tensors="pt")
    image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

    # generate the caption
    generated_caption = multimodal_model.generate(image, max_new_tokens=120)

    # save the caption
    with open(f"web_image_{idx}_caption.txt", "w") as f:
        f.write(f"Meta: {cartoon['meta']}\n")
        f.write(f"Generated Caption: {generated_caption}\n")

