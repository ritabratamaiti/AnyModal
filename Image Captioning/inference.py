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
from huggingface_hub import hf_hub_download

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE', 
    use_peft=False,
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "jmhessel/newyorker_caption_contest"

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)

cartoon_ds = vision.ImageDataset(dataset_name, image_processor, name = 'explanation', split = 'test')


# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Initialize MultiModalModel
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    input_start_token='<|imstart|>',
    input_end_token='<|imend|>',
    prompt_text="The description of the given New Yorker cartoon is: ")

if not os.path.exists("image_captioning_model"):
    os.makedirs("image_captioning_model")

hf_hub_download("AnyModal/VLM_Cartoon_Caption", filename="input_tokenizer.pt", local_dir="image_captioning_model")

# Load the model
multimodal_model._load_model("image_captioning_model")

# Generate captions for a few images and plot the images and save captions in txt file
import matplotlib.pyplot as plt

multimodal_model.eval()

for _ in range(5):
    sample_idx = np.random.randint(len(cartoon_ds))
    sample = cartoon_ds[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120, temperature = 0.2)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"temp/image_{sample_idx}.png")

    with open(f"temp/image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")




# daily_cartoons = [
#     {
#         'url': 'https://media.newyorker.com/cartoons/673756a82dadd75478bae26d/master/w_1920,c_limit/A60841.jpg',
#         'meta': 'Daily Cartoon: Friday, November 15th “You think they know something we don’t?”'
#     },
#     {
#         'url': 'https://media.newyorker.com/cartoons/6733645d1a159d6a06220768/master/w_1920,c_limit/A60826.jpg',
#         'meta': 'Daily Cartoon: Tuesday, November 12th ”I know it’s early, but it makes me believe we’ll make it to Christmas.”'
#     },
#     {
#         'url': 'https://media.newyorker.com/cartoons/672e2350db031fcd90d0cffc/master/w_1920,c_limit/A60823.jpg',
#         'meta': 'Daily Cartoon: Friday, November 8th Guess it’s that time again.'
#     },
#     {
#         'url': 'https://media.newyorker.com/cartoons/672cf4e877f87e5ffe6c2442/master/w_1920,c_limit/A60783.jpg',
#         'meta': 'Daily Cartoon: Thursday, November 7th “No one’s leaving until we can get them to agree on who to blame for this.”'
#     },
#     {
#         'url': 'https://media.newyorker.com/cartoons/672b83a71b813a89d3b0b4a7/master/w_1920,c_limit/A60820.jpg',
#         'meta': 'Daily Cartoon: Wednesday, November 6th Blown away.'
#     },
#     {
#       'url': 'https://media.newyorker.com/cartoons/6741068bdf4c1509157dd5ce/master/pass/241202_a28373_924.png',
#       'meta': ''
#     }
# ]


# # Generate captions for the daily cartoons
# for idx, cartoon in enumerate(daily_cartoons):
#     # download the image
#     response = requests.get(cartoon['url'])
#     img = Image.open(BytesIO(response.content))
#     img = img.convert('RGB')

#     # save the image
#     img.save(f"daily_cartoon_{idx}.png")

#     # process the image
#     image = image_processor(img, return_tensors="pt")
#     image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

#     # generate the caption
#     generated_caption = multimodal_model.generate(image, max_new_tokens=120)

#     # save the caption
#     with open(f"daily_cartoon_{idx}_caption.txt", "w") as f:
#         f.write(f"Meta: {cartoon['meta']}\n")
#         f.write(f"Generated Caption: {generated_caption}\n")

