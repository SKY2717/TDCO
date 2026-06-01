import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Optional

from models import build_model, build_mtl_model
from utils import load_checkpoint
from config import get_config

class MTLoRAGradCAM:
    """
    Grad-CAM visualization tool specifically designed for MTLoRA, 
    compatible with MultiTaskSwin architecture, and supporting the 
    visualization of multiple task decoders separately
    """
    def __init__(self, model, tasks: List[str], device='cuda'):

        
        self.model = model
        self.model.eval()
        self.tasks = tasks
        self.device = device
        self.gradients = {}
        self.activations = {}
        
    def _register_hooks(self, task_name: str):


        decoder = self.model.decoders.decoders[task_name]
        target_layer = decoder.last_layer[-1]  
        
        def forward_hook(module, input, output):
            self.activations[task_name] = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients[task_name] = grad_output[0].detach()
            
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        
        return [handle_fwd, handle_bwd]
    
    def generate_cam(self, input_image: torch.Tensor, task_name: str, 
                     target_category: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM for the specified task

        """

        self.model.zero_grad()
        self.gradients = {}
        self.activations = {}
        

        hooks = self._register_hooks(task_name)
        

        with torch.cuda.amp.autocast(enabled=False):  
            outputs = self.model(input_image)
        
        if task_name not in outputs:
            raise ValueError(f"task {task_name} is not in the model output. Available tasks: {list(outputs.keys())}")
            
        task_output = outputs[task_name]  # [1, num_classes, H, W]
        

        if target_category is not None:

            score = task_output[:, target_category, :, :].mean()
        else:

            score = task_output.max(dim=1)[0].mean()
        

        score.backward(retain_graph=True)
        

        for hook in hooks:
            hook.remove()
        

        gradients = self.gradients[task_name]  # [1, C, H', W']
        activations = self.activations[task_name]  # [1, C, H', W']
        

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        

        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)
        

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        

        cam = F.interpolate(cam, size=(input_image.shape[2], input_image.shape[3]), 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()
    
    def visualize_all_tasks(self, image_path: str, save_path: str = 'mtlora_gradcam.png',
                           img_size: int = 512):

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        

        denorm = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
        ])
        img_display = denorm(input_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        

        n_tasks = len(self.tasks)
        cols = min(n_tasks + 1, 4)
        rows = (n_tasks + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        

        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        for idx, task_name in enumerate(self.tasks):
            print(f"Generating Grad-CAM for task: {task_name}")
            
            try:
                cam = self.generate_cam(input_tensor, task_name)
                

                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = np.float32(heatmap) / 255
                

                superimposed = img_display * 0.6 + heatmap * 0.4
                superimposed = np.clip(superimposed, 0, 1)
                
                axes[idx + 1].imshow(superimposed)
                axes[idx + 1].set_title(f'{task_name}\nAttention Map')
                axes[idx + 1].axis('off')
                
            except Exception as e:
                print(f"Error processing {task_name}: {e}")
                import traceback
                traceback.print_exc()
                axes[idx + 1].text(0.5, 0.5, f'Error:\n{str(e)}', 
                                  ha='center', va='center', fontsize=8)
                axes[idx + 1].axis('off')
        

        for idx in range(len(self.tasks) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.show()


def load_mtlora_model_for_gradcam(config, checkpoint_path: str, device='cuda'):

    model = build_model(config)

    if config.MTL:
        model = build_mtl_model(model, config)
    

    model.to(device)
    

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
            
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)} keys")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} keys")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.eval()
    return model



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser('MTLoRA Grad-CAM Visualization')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--image', type=str, required=True, help='input image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint')
    parser.add_argument('--save', type=str, default='mtlora_gradcam.png', help='output path')
    parser.add_argument('--opts', help="Modify config options", default=None, nargs='+')
    

    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel (compatibility)')
    

    parser.add_argument('--batch-size', type=int, default=None, help="batch size")
    parser.add_argument('--data-path', type=str, default=None, help='path to dataset')
    parser.add_argument('--pascal', type=str, default=None, help='path to PASCAL dataset')
    parser.add_argument('--nyud', type=str, default=None, help='path to NYUD dataset')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained weights')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--throughput', action='store_true', help='test throughput')
    

    parser.add_argument('--tasks', type=str, default='semseg,normals,sal,human_parts',
                        help='List of tasks to visualize, e.g., semseg,normals,sal,human_parts')
    
    args = parser.parse_args()
    

    if not args.pascal and not args.nyud:

        dummy_path = "/tmp/pascal_context_dummy"
        os.makedirs(dummy_path, exist_ok=True)
        args.pascal = dummy_path
        print(f"Note: Using dummy Pascal path for config validation: {dummy_path}")
        print("      (This is normal for Grad-CAM visualization, dataset will not be loaded)")
    

    config = get_config(args)
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    print("Loading MTLoRA model...")
    model = load_mtlora_model_for_gradcam(config, args.checkpoint, device)
    

    if not config.MTL:
        raise ValueError("This script only supports MTL models")
    
    tasks = list(model.decoders.decoders.keys())
    print(f"Tasks detected from model: {tasks}")
    

    gradcam = MTLoRAGradCAM(model, tasks, device)
    

    print("Generating Grad-CAM visualizations...")
    img_size = config.DATA.IMG_SIZE if hasattr(config.DATA, 'IMG_SIZE') else 512
    gradcam.visualize_all_tasks(args.image, args.save, img_size=img_size)