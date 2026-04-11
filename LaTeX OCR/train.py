import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
from torch.amp import GradScaler

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE',   
    use_peft=True
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Dataset configuration
dataset_name = "unsloth/LaTeX_OCR"

# Load vision model components
# image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/siglip-so400m-patch14-384', use_peft=False)

# Load dataset
train_dataset = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'train')
val_dataset = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'test')

# get subset of the dataset
subset_ratio = 0.2

train_dataset = torch.utils.data.Subset(train_dataset, range(int(subset_ratio * len(train_dataset))))
val_dataset = torch.utils.data.Subset(val_dataset, range(int(subset_ratio * len(val_dataset))))


# DataLoader configuration
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_size = len(train_loader)
val_size = len(val_loader)

print(f"Train size: {train_size} | Validation size: {val_size}")

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

# Training configuration
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
optimizer.train()

# Scaler
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_losses.append(loss.item())
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())
        
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Decode a random validation sample
        for _ in range(5):
            sample_idx = np.random.randint(len(val_dataset))
            sample = val_dataset[sample_idx]
            print("Actual LaTeX: ", sample['text'])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                print("Generated LaTeX: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
            
    multimodal_model.train()
    os.makedirs(f"latex_ocr_{epoch+1}", exist_ok=True)
    multimodal_model._save_model(f"latex_ocr_{epoch+1}")


# evaluate on test set
# test_dataset = vision.ImageDataset(dataset_name, image_processor, name = None, split = 'test')
# test_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)

multimodal_model.eval()
test_losses = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Test", leave=False):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        test_losses.append(loss.item())
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Decode a random test sample
    for _ in range(5):
        sample_idx = np.random.randint(len(val_dataset))
        sample = val_dataset[sample_idx]
        print("Actual LaTeX: ", sample['text'])
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            print("Generated LaTeX: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
