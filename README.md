# ML-PROJECT
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow for displaying images in Colab

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image from file
image_path = '/content/drive/MyDrive/images.jpeg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to grayscale (Haar cascades work better on grayscale images)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output using cv2_imshow instead of cv2.imshow
cv2_imshow(image) # Use cv2_imshow to display the image in Colab
cv2.waitKey(0)
