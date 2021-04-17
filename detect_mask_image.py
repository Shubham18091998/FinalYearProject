# import the necessary packages
import warnings
warnings.filterwarnings(action="ignore")
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from digi.xbee.devices import XBeeDevice
import sys

# load our serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join(["./face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["./face_detector",
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] Loading face mask detector model...")
model = load_model("./mask_detector.model")

# load the input image from disk, clone it, and grab the image spatial dimensions
cap = cv2.VideoCapture(0)  # "0" for system webcam and "1" for external webcam
ret, image = cap.read()
orig = image.copy()
(h, w) = image.shape[:2]

mask_detected = 0  # Initialising variable to store mask detected or not
device = XBeeDevice("COM1", 9600)  # Instantiating a generic XBee device

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] Computing face detections...")
net.setInput(blob)
detections = net.forward()


# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the detection
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # ensure the bounding boxes fall within the dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face through the model to determine if the face has a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        mask_detected = 1 if label == "Mask" else 0
        # print(mask_detected)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

# Warning to wear mask
if mask_detected == 0:
    print("Please wear mask!")
    sys.exit()

# Sending data to Arduino
device.open()
device.send_data_broadcast(str(mask_detected))
device.close()

# Receiving data from Arduino
device.open()
xbee_msg = device.read_data(timeout=300)
temp_res = xbee_msg.data
device.close()
# print(chr(temp_res[0]))

# Checking mask and temperature to allow access
if mask_detected and int(chr(temp_res[0])):
    print("Access granted!")
else:
    print("Access not granted!")
