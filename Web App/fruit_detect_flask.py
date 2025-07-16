from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import time
import re

app = Flask(__name__)

# Initialize the object detection model
model = 'model_not_quantized_metadata.tflite' # Non Quantized but more accurate model
#model = 'model_quantized_edgetpu_metadata.tflite' # Quantized but less accurate model
num_threads = 4
enable_edgetpu = False # Uncomment if using Non Quantized Model
#enable_edgetpu = True # Uncomment if using Quantized Model

base_options = core.BaseOptions(
    file_name=model, num_threads=num_threads) # Uncomment if Non Quantized model is used
detection_options = processor.DetectionOptions(
    max_results=10, score_threshold=0.85) # Uncomment if Non Quantized model is used
#base_options = core.BaseOptions(
#    file_name=model, use_coral=enable_edgetpu, num_threads=num_threads) # Uncomment if Quantized model is used
#detection_options = processor.DetectionOptions(
#    max_results=10, score_threshold=0.55) # Uncomment if Quantized model is used
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Start capturing video input from the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.start()

input_width = picam2.preview_configuration.main.size[0]
input_height = picam2.preview_configuration.main.size[1]

# Initialize counters
pir_semirotten_count = 0
jeruk_semirotten_count = 0
display_pir_semirotten_count = 0
display_jeruk_semirotten_count = 0

def generate_frames():
    global pir_semirotten_count, jeruk_semirotten_count, display_pir_semirotten_count, display_jeruk_semirotten_count
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    while True:
        # Reset detection counters
        pir_semirotten_count = 0
        jeruk_semirotten_count = 0

        # Capture frame
        start_capture = time.time()
        image = picam2.capture_array()
        if image is None:
            print("Failed to capture image")
            continue
        capture_time = time.time() - start_capture
        
        # Resize the frame to the model's input dimensions
        image_resized = cv2.resize(image, (input_width, input_height))

        # Convert the image from BGR to RGB since by default OpenCV reads images in BGR format
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        input_tensor = vision.TensorImage.create_from_array(image_rgb)

        # Run object detection
        start_inference = time.time()
        detection_result = detector.detect(input_tensor)
        inference_time = time.time() - start_inference
        
        # Print Debug Info
        print(f"Capture time: {capture_time:.3f}s, Inference time: {inference_time:.3f}s")
        
        # Manually draw bounding boxes and text with custom colors
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            
            # Define custom colors
            bbox_color = (0, 255, 0)  # Green color for bounding box
            text_color = (255, 0, 0)  # Blue color for text
            
            # Draw bounding box
            cv2.rectangle(image_resized, start_point, end_point, bbox_color, 2)
            
            # Prepare text
            detection.categories[0].category_name = re.sub(r'[^\x20-\x7E]', '', detection.categories[0].category_name)
            class_name = detection.categories[0].category_name
            confidence = detection.categories[0].score
            text = f"{class_name}: {confidence:.2f}"
            
            # Put text on the image
            cv2.putText(image_resized, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            # Update detection counters
            if class_name == "Pir Semirotten":
                pir_semirotten_count += 1
                print(f"Pir Semirotten Count: {pir_semirotten_count}")  # Debug statement
            elif class_name == "Jeruk Semirotten":
                jeruk_semirotten_count += 1
                print(f"Jeruk Semirotten Count: {jeruk_semirotten_count}")  # Debug statement
        
        # Update display counters after detection is complete
        display_pir_semirotten_count = pir_semirotten_count
        display_jeruk_semirotten_count = jeruk_semirotten_count

        # Calculate average FPS every second
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0: # If 1 second has passed
            fps = frame_count / elapsed_time
            # print(f"FPS: {fps:.2f}")
            frame_count = 0 # Reset frame count
            start_time = current_time # Reset start time for next second
        # Overlay FPS on the image
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(image_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', image_resized)
        if not ret:
            print("Failed to encode frame")
            continue
        
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    global display_pir_semirotten_count, display_jeruk_semirotten_count
    print(f"Returning counts: Pir Semirotten: {display_pir_semirotten_count}, Jeruk Semirotten: {display_jeruk_semirotten_count}")  # Debug statement
    return jsonify(pir_semirotten=display_pir_semirotten_count, jeruk_semirotten=display_jeruk_semirotten_count)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
