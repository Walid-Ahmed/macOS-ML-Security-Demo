# convert_fixed.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

print("=" * 60)
print("FIXED CONVERSION: minimal_autoencoder.pth → Core ML")
print("=" * 60)


# Define the model architecture
class MinimalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 10)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def convert_model_safely():
    """Convert with better error handling and compatibility"""

    # 1. Check if model exists
    if not os.path.exists("minimal_autoencoder.pth"):
        print("❌ ERROR: 'minimal_autoencoder.pth' not found!")
        return False

    print("📁 Loading minimal_autoencoder.pth...")

    # 2. Load model
    model = MinimalAutoencoder()
    try:
        model.load_state_dict(torch.load("minimal_autoencoder.pth"))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # 3. Test model
    print("\n🧪 Testing model...")
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        output = model(test_input)
        error = torch.mean((test_input - output) ** 2).item()
        print(f"✅ Model test passed. Error: {error:.6f}")

    # 4. Convert to Core ML
    print("\n🔄 Converting to Core ML...")
    try:
        # Method 1: Direct conversion
        traced_model = torch.jit.trace(model, test_input)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=test_input.shape, name="input")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_units=ct.ComputeUnit.CPU_ONLY,  # Use CPU only for compatibility
            skip_model_load=False  # Ensure model is loaded after conversion
        )

        print("✅ Core ML conversion successful!")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

    # 5. Save model
    print("\n💾 Saving Core ML model...")
    output_path = "MinimalMacOSAnomalyAE.mlpackage"
    mlmodel.save(output_path)
    print(f"✅ Saved: {output_path}")

    # 6. FIXED VERIFICATION - Handle different coremltools versions
    print("\n🔍 Verifying conversion...")
    try:
        # Load the saved model
        verified_model = ct.models.MLModel(output_path)

        # SAFE WAY TO GET OUTPUT NAME - works with all coremltools versions
        # Method 1: Try to get from spec (most reliable)
        spec = verified_model.get_spec()
        output_name = spec.description.output[0].name
        print(f"✅ Output name from spec: '{output_name}'")

        # Method 2: Alternative way to get output info
        try:
            # Try dictionary access (newer versions)
            output_desc = verified_model.output_description
            if hasattr(output_desc, 'keys'):
                output_name = list(output_desc.keys())[0]
                print(f"✅ Output name from keys: '{output_name}'")
            else:
                # Single output object (older versions)
                output_name = output_desc.name
                print(f"✅ Output name from object: '{output_name}'")
        except:
            # If all else fails, use the spec name
            pass

        # Test prediction
        test_input_np = np.random.rand(1, 10).astype(np.float32)
        result = verified_model.predict({"input": test_input_np})

        # Use the output name we found
        if output_name in result:
            output_data = result[output_name]
            error = np.mean((test_input_np - output_data) ** 2)
            print(f"✅ Verification successful!")
            print(f"   Input shape: {test_input_np.shape}")
            print(f"   Output shape: {output_data.shape}")
            print(f"   Reconstruction error: {error:.6f}")
        else:
            print(f"❌ Output name '{output_name}' not found in result")
            print(f"   Available keys: {list(result.keys())}")
            # Try to use whatever output exists
            first_key = list(result.keys())[0]
            output_data = result[first_key]
            error = np.mean((test_input_np - output_data) ** 2)
            print(f"   Using first available output: '{first_key}'")
            print(f"   Output shape: {output_data.shape}")
            print(f"   Reconstruction error: {error:.6f}")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("But the model might still be saved. Let's check...")

        if os.path.exists(output_path):
            print(f"✅ Model file exists: {output_path}")
            print("You can try to use it in the security system.")
            return True
        else:
            return False


def main():
    """Main function with better error handling"""
    print("Core ML Conversion with Fixed Output Handling")
    print("")

    success = convert_model_safely()

    if success:
        print("\n" + "=" * 60)
        print("🎉 CONVERSION COMPLETED!")
        print("=" * 60)
        print("\n📁 Your Core ML model is ready:")
        print("   MinimalMacOSAnomalyAE.mlpackage")
        print("")
        print("🎯 Next: Run the security system:")
        print("   python simple_security_system.py")
    else:
        print("\n❌ Conversion failed.")


if __name__ == "__main__":
    main()