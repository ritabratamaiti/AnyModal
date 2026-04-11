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
    quantized = False,
    use_peft = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

llm_model.to(device)

# Dataset configuration
dataset_name = "unsloth/LaTeX_OCR"

# Load vision model components
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

ds = vision.TestDataset(dataset_name, image_processor, name = None, split = 'test')


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
    prompt_text="The latex expression of the equation in the image is:")

if not os.path.exists("latex_ocr"):
    os.makedirs("latex_ocr")

snapshot_download("AnyModal/latex-ocr-Llama-3.2-1B", local_dir="latex_ocr")

multimodal_model._load_model('latex_ocr')

# Generate captions for a few images and plot the images and save captions in txt file
multimodal_model.eval()

os.makedirs("temp", exist_ok=True)

for _ in range(5):
    sample_idx = np.random.randint(len(ds))
    sample = ds[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120, do_sample = True, num_beams = 3)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"temp/image_{sample_idx}.png")

    with open(f"temp/image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")




imgs = [
    {
        'url': 'https://datasets-server.huggingface.co/cached-assets/linxy/LaTeX_OCR/--/89aa6e447dd7afb4dec927af549df766539b6f9c/--/human_handwrite_print/train/0/image/image.jpg?Expires=1733310855&Signature=ILkAdXeXZ2K2ouUu8PjTfVi7BKSTQax1hNT04EYi2PrJmuewAM-JxAkC5VuPINekFCkbWqNf0V48FeVRylXuhHT1SQwfF7OXk~MwZ50hefzgr5E9HYqocIs5KgVNi8lAw6WgcQ~2MYKe9Rufy2lIFzchr0BfWqnbL1tmJXOlMjSKL78mb7vihoffoLyXcPpAQy2p0EjmZUvUlKR71wuQFtlT4lw5UOdPzk0oUNFvXt~E~42RC~5lhJMtmsYBm5KznMNTwoEiauHXnWu~QO14Z0ypgCOt1~mCDyILyKZCVhVcl6LHtstrk70-Id8djnEgQueQEvXbL~7zvoDgXEU2rw__&Key-Pair-Id=K3EI6M078Z3AC3',
        'meta': 'z _ { 1 } = r _ { 1 } ( \cos \theta _ { 1 } + i \sin \theta _ { 1 } )'
    },
    {
        'url': 'https://datasets-server.huggingface.co/cached-assets/linxy/LaTeX_OCR/--/89aa6e447dd7afb4dec927af549df766539b6f9c/--/human_handwrite_print/train/6/image/image.jpg?Expires=1733300239&Signature=wDz9TVpqHYeVq8tiM~xziv8k2-QxC5qIF1Ph4pJZpQV4xJfbMzA4Polzy7jMFXwkfJCET48cWTXLj9OE6t7fk2k-J9XIszVOBPfBCRn87WNCN3yEe0EhMIwLjMvg4JqFHel4LwkMLA6aMvg5pi9wgeZliZcA3smwGZecwZt5JLKNmo63v90nc695RRJPe1Tkko8IZuFp9WFR4oGQegdIlbno6I-iX-ZNXvdKrafq0-9KeQE~E2nOTwptJ5JEbZuI9E1a5bXEMufvw2v-87mLirEPptW2Zs1TOUojixQwSfnpCa6MgNuT3~92gs~s~ItTR2sc6UHm46HLAW5Ws7WcbQ__&Key-Pair-Id=K3EI6M078Z3AC3',
        'meta': '\frac { \tan \alpha - \tan \beta } { 1 + \tan \alpha \tan \beta }'
    }
]


# Generate captions for the daily cartoons
for idx, img in enumerate(imgs):
    # download the image
    response = requests.get(img['url'])
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')

    # save the image
    img.save(f"web_image_{idx}.png")

    # process the image
    image = image_processor(images = img, return_tensors="pt")
    image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

    # generate the caption
    generated_caption = multimodal_model.generate(image, max_new_tokens=120)

    # save the caption
    with open(f"web_image_{idx}_caption.txt", "w") as f:
        f.write(f"Meta: {img['meta']}\n")
        f.write(f"Generated Caption: {generated_caption}\n")


