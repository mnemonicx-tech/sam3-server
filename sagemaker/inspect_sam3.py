import sys
import inspect

print("🔍 Inspecting installed SAM 3 headers...")

try:
    import sam3
    print("✅ Successfully imported 'sam3'")
    print(f"📁 Package location: {sam3.__file__}")
    print(f"📜 Top-level contents: {dir(sam3)}")

    # Check for build functions
    if hasattr(sam3, "build_sam3"):
        print("✅ Found 'build_sam3'")
    
    # Check for predictor
    if hasattr(sam3, "Sam3Predictor") or hasattr(sam3, "SAM3Predictor"):
        print("✅ Found Predictor class")
    
    # Prrint recursive search for "predict" methods
    print("\n🔍 Searching for 'predict' method signature...")
    
    def find_predict(module, path="sam3"):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if name.startswith("Sam") or name.startswith("SAM"):
                    # Check methods
                    for m_name, m_obj in inspect.getmembers(obj):
                        if m_name == "predict":
                             print(f"  ➡️ Found {path}.{name}.predict signature: {inspect.signature(m_obj)}")
            elif inspect.ismodule(obj) and name.startswith("sam3"):
                 # Recurse (limit depth ideally, but basic is fine)
                 pass

    find_predict(sam3)

except ImportError:
    print("❌ Could not import 'sam3'. Trying 'segment_anything_3'...")
    try:
        import segment_anything_3
        print("✅ Successfully imported 'segment_anything_3'")
        print(f"📜 Contents: {dir(segment_anything_3)}")
    except ImportError:
        print("❌ Could not import 'sam3' or 'segment_anything_3'. Check installation.")

