import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.model_loader import load_model
from utils.image_utils import preprocess_image, postprocess_image
from config import OUTPUT_DIR
import uuid

def process_image(input_image, model_id):
    """
    Process an input image with the specified model.

    Args:
        input_image (PIL.Image): Input image to process
        model_id (str): Model identifier

    Returns:
        str: Path to the saved output image
    """
    try:
        # Generate a unique filename for the output
        output_filename = f"{model_id}_{uuid.uuid4().hex}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Load the model
        print(f"Loading model: {model_id}")
        model = load_model(model_id)
        model.eval() # Set model to evaluation mode

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.to(device) # Move model to the correct device

        output_image = None # Initialize output PIL image

        # Perform inference based on the model type
        if model_id == "spectroformer":
            print("Processing with Spectroformer...")

            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            
            input_image = input_image.convert("RGB").resize((512,512), Image.BICUBIC)

            input_tensor = transform(input_image).unsqueeze(0).to(device)
            print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")


            with torch.no_grad():
                print("Running model inference...")
                raw_output = model(input_tensor)


                if isinstance(raw_output, (list, tuple)):
                    output_tensor_gpu = raw_output[0]
                else:
                    output_tensor_gpu = raw_output 

                print(f"Raw output tensor shape (GPU): {output_tensor_gpu.shape}")


                output_tensor_cpu = output_tensor_gpu.detach().squeeze(0).cpu()
                print(f"Processed output tensor shape (CPU): {output_tensor_cpu.shape}")

            output_tensor_cpu = output_tensor_cpu * 0.5 + 0.5

            output_tensor_cpu = torch.clamp(output_tensor_cpu, 0, 1)

            # Convert tensor to PIL Image
            to_pil_transform = transforms.ToPILImage()
            output_image = to_pil_transform(output_tensor_cpu)
            print("Converted output tensor to PIL image.")
        
        elif model_id == "phaseformer":
            print("Processing with Spectroformer...")

            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            
            input_image = input_image.convert("RGB").resize((512,512), Image.BICUBIC)

            input_tensor = transform(input_image).unsqueeze(0).to(device)
            print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")


            with torch.no_grad():
                print("Running model inference...")
                raw_output = model(input_tensor)


                if isinstance(raw_output, (list, tuple)):
                    output_tensor_gpu = raw_output[0]
                else:
                    output_tensor_gpu = raw_output 

                print(f"Raw output tensor shape (GPU): {output_tensor_gpu.shape}")


                output_tensor_cpu = output_tensor_gpu.detach().squeeze(0).cpu()
                print(f"Processed output tensor shape (CPU): {output_tensor_cpu.shape}")

            output_tensor_cpu = output_tensor_cpu * 0.5 + 0.5

            output_tensor_cpu = torch.clamp(output_tensor_cpu, 0, 1)

            # Convert tensor to PIL Image
            to_pil_transform = transforms.ToPILImage()
            output_image = to_pil_transform(output_tensor_cpu)
            print("Converted output tensor to PIL image.")

        else:
            raise ValueError(f"Inference not implemented for model: {model_id}")

        # Save the final output image if it was generated
        if output_image:
            print(f"Saving output image to: {output_path}")
            output_image.save(output_path)
            return output_path
        else:
             raise RuntimeError(f"Output image was not generated for model {model_id}.")

    except Exception as e:
        print(f"Error during inference for model {model_id}: {str(e)}")

        raise Exception(f"Error during inference for model {model_id}: {str(e)}")

    
def run_spectroformer(input_tensor, model):
    final_l = model(input_tensor)[0]
    final_l = final_l.detach().squeeze(0).cpu()
    
    return final_l

def simulate_style_transfer(input_tensor, model):
    """Simulate a style transfer output for demonstration."""
    # This is a placeholder for actual style transfer implementation
    # In a real app, you would use your trained style transfer model here
    output = model(input_tensor)['out'][0]
    # Normalize to [0, 1] range for visualization
    output = (output - output.min()) / (output.max() - output.min())
    return output

def simulate_super_resolution(input_tensor, model):
    """Simulate a super resolution output for demonstration."""
    # This is a placeholder for actual super resolution implementation
    # In a real app, you would use your trained super resolution model here
    output = model(input_tensor)['out'][0]
    # Normalize to [0, 1] range for visualization
    output = (output - output.min()) / (output.max() - output.min())
    return output