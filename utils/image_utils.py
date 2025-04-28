import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from config import TEMP_DIR

def save_uploaded_image(image, filename):
    """
    Save the uploaded image to a temporary directory.
    
    Args:
        image (PIL.Image): Image to save
        filename (str): Original filename
        
    Returns:
        str: Path to the saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Get file extension
    ext = os.path.splitext(filename)[1].lower()
    
    # Save image to temp directory
    path = os.path.join(TEMP_DIR, f"input{ext}")
    image.save(path)
    
    return path

def preprocess_image(image, model_id):
    """
    Preprocess an image for the specified model.
    
    Args:
        image (PIL.Image): Image to preprocess
        model_id (str): Model identifier
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if model_id == "style_transfer_model":
        # Preprocess for style transfer
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    elif model_id == "segmentation_model":
        # Preprocess for segmentation
        transform = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    elif model_id == "super_resolution_model":
        # Preprocess for super resolution
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    else:
        # Default preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Apply the transformations
    input_tensor = transform(image)
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor

def postprocess_image(tensor, model_id):
    """
    Convert a tensor to a PIL Image after model processing.
    
    Args:
        tensor (torch.Tensor): Output tensor from the model
        model_id (str): Model identifier
        
    Returns:
        PIL.Image: Processed image
    """
    # Move tensor to CPU if it's on GPU
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    if model_id == "segmentation_model":
        # Create a colorful segmentation map
        colors = torch.tensor([
            [0, 0, 0],       # background
            [128, 0, 0],     # aeroplane
            [0, 128, 0],     # bicycle
            [128, 128, 0],   # bird
            [0, 0, 128],     # boat
            [128, 0, 128],   # bottle
            [0, 128, 128],   # bus
            [128, 128, 128], # car
            [64, 0, 0],      # cat
            [192, 0, 0],     # chair
            [64, 128, 0],    # cow
            [192, 128, 0],   # dining table
            [64, 0, 128],    # dog
            [192, 0, 128],   # horse
            [64, 128, 128],  # motorbike
            [192, 128, 128], # person
            [0, 64, 0],      # potted plant
            [128, 64, 0],    # sheep
            [0, 192, 0],     # sofa
            [128, 192, 0],   # train
            [0, 64, 128]     # tv/monitor
        ], dtype=torch.float) / 255.0
        
        # Convert segmentation map to RGB image
        output_image = Image.fromarray(
            tensor.byte().numpy() % len(colors)
        ).convert('P')
        palette = []
        for color in colors:
            palette.extend((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
        output_image.putpalette(palette)
        output_image = output_image.convert('RGB')
        
    else:
        # For style transfer and super resolution, convert tensor to image
        if len(tensor.shape) == 3:
            # Handle output with 3 channels (RGB)
            np_array = tensor.numpy().transpose(1, 2, 0)
            
            # Ensure values are in [0, 1] range
            np_array = np.clip(np_array, 0, 1)
            
            # Convert to 8-bit unsigned integers
            np_array = (np_array * 255).astype(np.uint8)
            
        else:
            # Handle single channel output
            np_array = tensor.numpy()
            
            # Ensure values are in [0, 1] range
            np_array = np.clip(np_array, 0, 1)
            
            # Convert to 8-bit unsigned integers
            np_array = (np_array * 255).astype(np.uint8)
        
        output_image = Image.fromarray(np_array)
    
    return output_image