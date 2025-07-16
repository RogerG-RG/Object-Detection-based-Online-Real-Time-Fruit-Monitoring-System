import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_9/exported-models/my_model/saved_model') # path to the SavedModel directory
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # Uncomment this line for Dynamic Range Quantization
tflite_model = converter.convert()


# Save the model.
with open('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model/saved_model.tflite', 'wb') as f:
  f.write(tflite_model)