import sys
import inspect

print("🔍 Inspecting Ultralytics for SAM 3 support...")

try:
    import ultralytics
    print(f"✅ Successfully imported 'ultralytics': {ultralytics.__version__}")
    
    from ultralytics import SAM
    print("✅ Found 'SAM' class in ultralytics")
    
    # Check if we can find SAM3 references
    import ultralytics.models.sam
    print(f"📜 ultralytics.models.sam contents: {dir(ultralytics.models.sam)}")

    # Try to find specific predictor classes
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
        print("✅ Found 'SAM3SemanticPredictor'")
    except ImportError:
        print("❌ Could not import 'SAM3SemanticPredictor' directly")

    # Inspect the SAM object to see if it handles 'sam3' weights
    print("\nℹ️  To verify SAM 3 support, we usually just load the model:")
    print("   model = SAM('sam3_large.pt')")
    print("   results = model(source='image.jpg', bboxes=[...], labels=[...])")

except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure you have installed: pip install ultralytics>=8.3.237")

