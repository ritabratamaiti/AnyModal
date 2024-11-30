import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE'
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "Mozilla/coco-gpt4o"

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)


train_dataset = vision.ImageDataset(dataset_name, image_processor, split = 'train')
val_dataset = vision.ImageDataset(dataset_name, image_processor, split = 'validation')

train_size = len(train_dataset)
val_size = len(val_dataset)
print(f"Train size: {train_size}, Validation size: {val_size}")

# DataLoader configuration
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(train_size//10)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(torch.utils.data.Subset(val_dataset, range(val_size//10)), batch_size=batch_size, shuffle=True)


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
    prompt_text="The description of the given image is: ")

multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)

# Training configuration
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
optimizer.train()

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        logits, loss = multimodal_model(batch)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())
        
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Decode a random validation sample
        for _ in range(5):
            sample_idx = np.random.randint(len(val_dataset))
            sample = val_dataset[sample_idx]
            print("Actual Text: ", sample['text'])
            print("Generated Text: ", multimodal_model.generate(sample['input'], max_new_tokens=120))

    multimodal_model.train()

os.makedirs("image_captioning_model", exist_ok=True)

# Save the model
multimodal_model._save_model("image_captioning_model")

del(multimodal_model)

multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1),
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    input_start_token='<|imstart|>',
    input_end_token='<|imend|>',
    prompt_text="The description of the given image is: ")

# Load the model
multimodal_model._load_model("image_captioning_model")

# Generate captions for a few images and plot the images and save captions in txt file


multimodal_model.eval()

for _ in range(5):
    sample_idx = np.random.randint(len(val_dataset))
    sample = val_dataset[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"image_{sample_idx}.png")

    with open(f"image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")