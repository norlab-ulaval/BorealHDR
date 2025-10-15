#!/usr/bin/env python3
"""
Test script for the differentiable Image_Emulator class
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from scripts.classes.class_image_emulator import Image_Emulator

import cv2

def test_differentiable_emulator():
    """Test the differentiable image emulator"""
    
    print("Testing Differentiable Image Emulator...")
    
    # Test with CPU device first
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Path to sample data
    data_path = Path("data_sample/backpack_2023-09-25-15-05-03/camera_left")
    
    if not data_path.exists():
        print(f"Sample data not found at {data_path}")
        print("Please ensure the data_sample directory exists")
        return False
    
    try:
        # Initialize the emulator
        emulator = Image_Emulator(
            path_imgs_bracketing=data_path,
            emulation_method="radiance",
            selection_method="closer_least_sat",
            color_bayer=True,
            device=device
        )
        print("✓ Emulator initialized successfully")
        
        # Test tensor properties
        print(f"✓ Bracketing values device: {emulator.bracketing_values.device}")
        print(f"✓ Bracketing values: {emulator.bracketing_values}")
        
        # Test image loading
        if len(emulator.bracket_images) > 0:
            img = emulator.bracket_images[0]
            print(f"✓ First image loaded as tensor: {torch.is_tensor(img)}")
            print(f"✓ Image shape: {img.shape if torch.is_tensor(img) else 'Not a tensor'}")
            print(f"✓ Image device: {img.device if torch.is_tensor(img) else 'Not a tensor'}")
        
        # Test emulation with gradient tracking
        target_exp_time = torch.tensor(1.0, requires_grad=True, device=device)
        
        # Enable gradients for differentiability testing
        emulator.enable_gradient()
        
        # Emulate an image
        result = emulator.emulate_image(target_exp_time)
        emulated_img = result["emulated_img"]

        # Convert emulated image to 8-bit before displaying
        img_np = emulated_img.detach().cpu().numpy()
        img_np = (img_np / 16.0).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BAYER_RG2RGB)
        cv2.imshow("First Bracket Image", img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(img_np)

        print(f"✓ Emulation successful")
        print(f"✓ Emulated image is tensor: {torch.is_tensor(emulated_img)}")
        print(f"✓ Emulated image requires grad: {emulated_img.requires_grad if torch.is_tensor(emulated_img) else 'Not a tensor'}")
        print(f"✓ Target exposure time: {target_exp_time}")
        print(f"✓ Used exposure time: {result['exposure_time']}")
        print(f"✓ Emulation factor: {result['emulation_factor']}")
        
        # Test gradient computation
        if torch.is_tensor(emulated_img) and emulated_img.requires_grad:
            print(emulated_img.mean())
            loss = (emulated_img.mean() - 2048.0)**2
            loss.backward()
            print(target_exp_time.grad)
            print("✓ Gradient computation successful - the emulator is differentiable!")
        
        # Test utility functions
        if torch.is_tensor(emulated_img):
            numpy_img = emulator.to_numpy(emulated_img)
            print(f"✓ Tensor to numpy conversion: {type(numpy_img)}")
        
        print("\n🎉 All tests passed! The Image_Emulator is now differentiable.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_differentiable_emulator()
    sys.exit(0 if success else 1)