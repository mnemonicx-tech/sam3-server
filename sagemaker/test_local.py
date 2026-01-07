import os
import sys

# Mock Environment Variables
os.environ["INPUT_ROOT"] = "sample_data/input"
os.environ["OUTPUT_ROOT"] = "sample_data/output"

# Mock Paths (ensure these exist or are created)
os.makedirs("sample_data/input/bottomwear_men_jeans", exist_ok=True)

# Create dummy image if none exists
dummy_img_path = "sample_data/input/bottomwear_men_jeans/test_jeans.jpg"
if not os.path.exists(dummy_img_path):
    print("Creating dummy image for testing...")
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    img.save(dummy_img_path)

print("Starting Local Inference Test...")
print("Note: This requires 'groundingdino', 'segment_anything', 'torch' installed in your local environment.")
print("If running outside the Docker container, ensure you have these dependencies.")

try:
    import inference
    inference.main()
    print("\n✅ Test Completed Successfully!")
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("Please run this test inside the Docker container or install dependencies locally.")
except Exception as e:
    print(f"\n❌ Runtime Error: {e}")
