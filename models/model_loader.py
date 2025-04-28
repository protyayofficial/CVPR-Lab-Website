import os
import sys # <-- Import sys
import torch
import streamlit as st

models_dir = os.path.dirname(os.path.abspath(__file__))

# Cache the loaded models to prevent reloading
@st.cache_resource
def load_model(model_id):
    """
    Load a PyTorch model based on the model identifier.

    Args:
        model_id (str): Identifier of the model to load

    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    # --- Add models directory to sys.path ---
    original_sys_path = list(sys.path) # Store original path
    if models_dir not in sys.path:
        print(f"Temporarily adding {models_dir} to sys.path for torch.load")
        sys.path.insert(0, models_dir)
        added_path = True
    else:
        added_path = False
    # ---------------------------------------

    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_id == "spectroformer":
            print("Attempting torch.load for spectroformer.pth...")
            # Now torch.load should be able to find the 'model_without_CA' module
            spectroformer = torch.load("models/checkpoints/spectroformer.pth", map_location='cpu')
            print("torch.load successful.")
            spectroformer.to(device)
            spectroformer.eval()
            return spectroformer
        
        elif model_id == "phaseformer":
            phaseformer = torch.load("models/checkpoints/phaseformer_UIEB.pth", map_location='cpu')
            phaseformer.to(device)
            phaseformer.eval()
            return phaseformer

        else:
            raise ValueError(f"Unknown model identifier: {model_id}")

    except ModuleNotFoundError as e:
         # Add more specific error logging
         print(f"Caught ModuleNotFoundError during torch.load: {e}")
         print(f"Current sys.path: {sys.path}")
         print("This usually means the model checkpoint was saved expecting the module")
         print(f"'{e.name}' to be available as a top-level module.")
         print(f"Tried adding '{models_dir}' to sys.path.")
         raise Exception(f"Failed to load model {model_id} due to missing module definition: {str(e)}")
    except Exception as e:
        # General error
        import traceback
        print(f"Generic error loading model {model_id}:")
        traceback.print_exc()
        raise Exception(f"Failed to load model {model_id}: {str(e)}")
    finally:
        # --- Clean up sys.path ---
        if added_path:
            try:
                sys.path.remove(models_dir)
                print(f"Removed {models_dir} from sys.path")
            except ValueError:
                pass # Should not happen if added_path is True, but be safe
        # -------------------------


