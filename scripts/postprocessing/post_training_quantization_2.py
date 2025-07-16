import tensorflow as tf
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter

image_path = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/images/train'
quant_images_list = glob.glob(image_path + '/*.jpg')

# interpreter = Interpreter(model_path='C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
height = 640
width = 640

def representative_data_gen():
  dataset_list = quant_images_list
  quant_num = 17500
  for i in range(quant_num):
    pick_me = random.choice(dataset_list)
    image = tf.io.read_file(pick_me)

    if pick_me.endswith('.jpg') or pick_me.endswith('.JPG'):
      image = tf.io.decode_jpeg(image, channels=3)
    elif pick_me.endswith('.png'):
      image = tf.io.decode_png(image, channels=3)
    elif pick_me.endswith('.bmp'):
      image = tf.io.decode_bmp(image, channels=3)

    image = tf.image.resize(image, [width, height])  # TO DO: Replace 300s with an automatic way of reading network input size
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)

# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input tensors to uint8 and output tensors to float32
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model/model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

quantized_interpreter = Interpreter(model_path='C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model/model_quantized.tflite')
input_type = quantized_interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = quantized_interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)