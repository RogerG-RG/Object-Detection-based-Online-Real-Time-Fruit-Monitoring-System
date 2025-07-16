# import tensorflow as tf
# import numpy as np
# import glob
# from PIL import Image

# def representative_dataset_gen():
#     folder = "C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/images/quant"
#     image_size = 640
#     raw_test_data = []

#     files = glob.glob(folder + '/*.jpg')
#     for file in files:
#         image = Image.open(file)
#         image = image.convert("RGB")
#         image = image.resize((image_size, image_size))
#         # Quantizing the image between 0 and 255
#         image = np.asarray(image).astype(np.float32) / 255.0
#         image = image[np.newaxis, :, :, :]
#         raw_test_data.append(image)

#     for data in raw_test_data:
#         yield [data]

# converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model')  # path to the SavedModel directory
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.allow_custom_ops = True
# converter.representative_dataset = representative_dataset_gen
# tflite_model = converter.convert()

# # Save the model.
# with open('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model/model_quantized.tflite', 'wb') as f:
#     f.write(tflite_model)


# import tensorflow as tf
# import numpy as np

# def format_image(image, image_size):
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, (image_size, image_size))
#     image = image[tf.newaxis, :]
#     return image

# raw_test_data = tf.data.Dataset.list_files('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_6/images/quant/*.jpg')
# test_batches = raw_test_data.batch(1)

# def representative_dataset_gen():
#     for input_value, _ in test_batches.take(100):
#         yield [input_value]

# converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model')  # path to the SavedModel directory
# converter.representative_dataset = representative_dataset_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# tflite_model = converter.convert()
# tflite_model_dir = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model/model_quantized.tflite'
# with open(tflite_model_dir, 'wb') as f:
#     f.write(tflite_model)


# import tensorflow as tf
# import numpy as np

# def representative_dataset_gen():
#     raw_test_data = tf.data.Dataset.list_files('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/images/quant/*.jpg')
#     test_batches = raw_test_data.batch(1)

#     for image_path in test_batches.take(100):
#         image = tf.io.read_file(image_path[0])
#         image = tf.image.decode_jpeg(image, channels=3)
#         image = tf.image.resize(image, [640, 640])
#         image = tf.cast(image, tf.float32) / 255.0
#         image = tf.expand_dims(image, 0)
#         yield [image]

# converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model')  # path to the SavedModel directory
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# # converter.experimental_new_quantizer = True  # Enable the new quantizer
# converter.representative_dataset = representative_dataset_gen

# tflite_model = converter.convert()
# tflite_model_dir = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model/model_quantized.tflite'
# with open(tflite_model_dir, 'wb') as f:
#     f.write(tflite_model)



import tensorflow as tf
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter

image_path = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/images/train'
quant_images_list = glob.glob(image_path + '/*.jpg')
# PATH_TO_MODEL = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_5/exported-models/my_model_4/saved_model'
# interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# height = input_details[0]['shape'][1]
# width = input_details[0]['shape'][2]

def representative_data_gen():
    dataset_list = quant_images_list
    quant_num = 10000
    for image in range(quant_num):
        image_path = random.choice(dataset_list)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [640, 640])
        image = tf.cast(image / 255, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

# def representative_data_gen():
#     for image in tf.data.Dataset.from_tensor_slices(quant_images_list).batch(1).take(100):
#         yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model/model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = Interpreter(model_path='C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_10/exported-models/my_model/saved_model/model_quantized.tflite')
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)