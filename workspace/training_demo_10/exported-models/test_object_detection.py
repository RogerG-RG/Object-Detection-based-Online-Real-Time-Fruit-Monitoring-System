# Import packages
import os
import cv2
import numpy as np
import glob
import random
import argparse
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt

def tflite_detect_images(modelpath, imgpath, min_conf=0.5, num_test_images=10, savepath='results', txt_only=False):
    # Grab filenames of all images in test folder
    images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

    # Load the label map into memory
    labels = ["Pear Semirotten", "Orange Semirotten"]  # Replace with your own label map

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Randomly select test images
    images_to_test = random.sample(images, num_test_images)

    # Loop over every image and perform detection
    for image_path in images_to_test:
        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e., if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                ymin = int(max(1, (boxes[i][0] * image.shape[0])))
                xmin = int(max(1, (boxes[i][1] * image.shape[1])))
                ymax = int(min(image.shape[0], (boxes[i][2] * image.shape[0])))
                xmax = int(min(image.shape[1], (boxes[i][3] * image.shape[1])))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 1)

                # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Get font size
                label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Draw label text

        # Save the image with detections
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        output_path = os.path.join(savepath, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f"Processed {image_path}")

        # Optionally, display the image
        # if not txt_only:
        #     plt.figure(figsize=(12, 8))
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', help='Path to the TFLite model file', required=True)
    parser.add_argument('--imgpath', help='Path to the folder containing test images', required=True)
    parser.add_argument('--min_conf', help='Minimum confidence threshold for displaying detected objects', type=float, default=0.5)
    parser.add_argument('--num_test_images', help='Number of test images to run inference on', type=int, default=10)
    parser.add_argument('--savepath', help='Path to save the output images with detections', default='results')
    parser.add_argument('--txt_only', help='Only save results to text files without displaying images', action='store_true')
    args = parser.parse_args()

    tflite_detect_images(args.modelpath, args.imgpath, args.min_conf, args.num_test_images, args.savepath, args.txt_only)

if __name__ == '__main__':
    main()