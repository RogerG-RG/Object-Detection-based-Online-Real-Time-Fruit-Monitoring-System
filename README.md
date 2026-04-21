# Object Detection-based Online Real-Time Fruit Monitoring System
## Project Description
This project is for an AI-Powered System to Monitor Pears and 3 different types of Oranges intended to be used on Supermarket Fruit Display with the purpose of detecting Pears and Oranges that start to go bad / Semirotten accessible through a VPN protected Web Application. The purpose of detecting the Semirotten fruit is to spot them early while the fruit is still fit for consumption so it could be separated and used early to minimize food waste. The system utilize the Raspberry Pi 4, Raspberry Pi Camera, and Google Coral USB Accelerator TPU (Optional) to handle the inference process.
## How it works
When fruit_detect_flask.py is run, it would initiate the Web App that the user could only access when connected to the same Tailscale account as the Raspberry Pi. The user could see the real time detection result in the form of a live feed that the system staarted inferencing after the Web App became accesible. There are two types of models provided which are:
- model_not_quantized_metadata.tflite
- model_quantized_metadata.tflite
The non-quantized model is only utilizing the Raspberry Pi's hardware to perform the inference process which means it is slow while the quantized model has been optimized and went through additional steps to allow it to utilize the Google Coral Edge TPU partially which helps speed up the inference process. However, based on testing, the non-quantized model's accuracy performed better compared to the quantized model. This could be due to the use of Post-training quantization method which could cause accuracy degradation compared to Quantize-aware training which I did not have access to at the time of making the project due to being unsupported in the Tensorflow version I was using.
## Our LinkedIn
- [Roger Gibson] (https://www.linkedin.com/in/roger-the-engineer/)
- [Michael Halim] (https://www.linkedin.com/in/michaelhalim03/)
