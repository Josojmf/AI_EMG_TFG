import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model

# File paths
OLD_MODEL_PATH = "cnn_FFT.keras"
NEW_MODEL_PATH = "cnn_FFT_fixed.h5"

try:
    print("🔄 Attempting to load the corrupted `.keras` model...")
    
    # Load the original model as a dictionary (without deserializing layers)
    model_data = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)

    # Extract the architecture as JSON
    print("📝 Extracting model architecture...")
    model_json = model_data.to_json()

    # Save JSON structure (for debugging)
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    print("🔄 Rebuilding model from JSON...")
    new_model = model_from_json(model_json)

    # Load weights from the broken model
    print("⚖️ Restoring weights...")
    new_model.set_weights(model_data.get_weights())

    # Save the fixed model as HDF5 (compatible format)
    print("💾 Saving fixed model...")
    new_model.save(NEW_MODEL_PATH)
    print(f"✅ Model successfully rebuilt and saved as: {NEW_MODEL_PATH}")

except Exception as e:
    print(f"❌ Critical error: {e}")
