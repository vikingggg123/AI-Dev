from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
               directory: str,
               model_name: str):
    
    target_dir_path = Path(directory)

    assert model_name.endswith(".pth")
    model_save_path = target_dir_path / model_name

    print(f"saving model {model_name} to the path: {model_save_path}")
    torch.save(model,
               f=model_save_path)

    pass