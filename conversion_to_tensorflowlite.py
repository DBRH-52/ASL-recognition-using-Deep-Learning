# conversion_to_tensorflowlite.py
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = 'asl_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Convert the TensorFlow Lite model to a C array
c_array_path = 'asl_model_data.h'

with open(tflite_model_path, 'rb') as f:
    tflite_model_data = f.read()

with open(c_array_path, 'w') as f:
    # Write C array header
    f.write('#ifndef ASL_MODEL_DATA_H\n')
    f.write('#define ASL_MODEL_DATA_H\n\n')

    # Write the C array declaration
    f.write('const unsigned char asl_model_data[] = {')

    # Write the hex values of each byte in the TensorFlow Lite model
    for byte in tflite_model_data:
        f.write(f'{hex(byte)}, ')

    # Complete the C array declaration
    f.write('};\n\n')
    f.write('#endif // ASL_MODEL_DATA_H\n')
