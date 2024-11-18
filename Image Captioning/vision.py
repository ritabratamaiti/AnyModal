from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
import datasets
from peft import get_peft_config, get_peft_model, LoraConfig

class ImageDataset(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, dataset_name, processor, name, split='train'):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        - split: str, specifies the dataset split (default: 'train').
        """
        self.dataset = datasets.load_dataset(dataset_name, name)[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        item = self.dataset[idx]
        image = item['image']

        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        rgb_val = image
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

        return {
            'input': image,
            'text': item['image_description'],
            'image': rgb_val
        }

class Projector(nn.Module):
    """
    Projector: A feedforward neural network for projecting feature embeddings to a target dimension.

    Attributes:
    - inp_layer: Input linear layer.
    - layers: Sequence of hidden layers.
    - dropout: Dropout applied between layers.
    - out_layer: Output linear layer.
    """

    def __init__(self, in_features, out_features, num_hidden=2):
        """
        Initializes the Projector.

        Parameters:
        - in_features: int, size of the input feature vector.
        - out_features: int, size of the output feature vector.
        - num_hidden: int, number of hidden layers (default: 2).
        """
        super(Projector, self).__init__()
        self.inp_layer = nn.Linear(in_features, out_features)
        self.layers = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_hidden)])
        self.dropout = nn.Dropout(0.1)
        self.out_layer = nn.Linear(out_features, out_features)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: torch.Tensor, input tensor.

        Returns:
        - torch.Tensor, output tensor.
        """
        x = self.inp_layer(x)
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        x = self.out_layer(x)
        return x

class VisionEncoder(nn.Module):
    """
    VisionEncoder: Wraps a vision model to extract hidden states as feature embeddings.

    Attributes:
    - model: Pre-trained vision model.
    - device: Torch device (GPU/CPU).
    """

    def __init__(self, model):
        """
        Initializes the VisionEncoder.

        Parameters:
        - model: nn.Module, pre-trained vision model.
        """
        super(VisionEncoder, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, inputs):
        """
        Forward pass to obtain feature embeddings.

        Parameters:
        - inputs: dict, preprocessed inputs compatible with the vision model.

        Returns:
        - torch.Tensor, last hidden state of the vision model.
        """
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Extract last hidden state

def get_image_encoder(model_name, use_peft=False):
    """
    Loads a vision model and its processor, optionally applying Parameter-Efficient Fine-Tuning (PEFT).

    Parameters:
    - model_name: str, name of the pre-trained vision model.
    - use_peft: bool, whether to apply PEFT (default: False).

    Returns:
    - processor: Image processor for pre-processing.
    - model: Pre-trained vision model.
    - hidden_size: int, size of the model's hidden layer.
    """
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    hidden_size = model.config.hidden_size
    
    if use_peft:
        
        
        peft_config = LoraConfig(
            task_type=None, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['dense']
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad = False

    return processor, model, hidden_size

if __name__ == '__main__':
    dataset_name = "AnaniyaX/indiana_uni_chest_x_ray"
    processor, model, hidden_size = get_image_encoder('google/vit-base-patch16-224')

    dataset = ImageDataset(dataset_name, processor)

    # Split dataset
    split_ratio = 0.8
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader setup
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder and projector
    vision_encoder = VisionEncoder(model)
    vision_projector = Projector(hidden_size, 768)

    for batch in train_loader:
        vision_embeddings = vision_encoder(batch['input'])
        print(vision_embeddings.shape)
        vision_tokens = vision_projector(vision_embeddings)
        print(vision_tokens.shape)
        break
