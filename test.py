from detect import *
from architecture import *
import mtcnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from PIL import Image, ImageTk
import tempfile
import os

# plotting the image
def plot_result_image(result_img, recognized_name, confidence_score):
    plt.imshow(result_img)
    title = f"{recognized_name} ({confidence_score:.2f}%)"
    plt.title(title)
    plt.show()

#taking the url as input from the user
webpage_url = input("Web page URL :")
# webpage_url = 'https://www.imdb.com/list/ls053501318/'  # Replace with your target webpage URL
images = extract_images_from_webpage(webpage_url)

#loading the facenet model, SVM and associated weights.
face_encoder = InceptionResNetV2()
path_m = "facenet_keras_weights.h5"
face_encoder.load_weights(path_m)
face_detector = mtcnn.MTCNN()
data = []
for i, img_array in enumerate(images):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    result_img, recognized_name, confidence_score = detect(img, face_detector, face_encoder)
    data.append([recognized_name, f"{confidence_score:.2f}"])
    plot_result_image(result_img, recognized_name, confidence_score)
    plot_result_image(img, recognized_name, confidence_score)

# Display recognized names and confidence scores
j = 1
for entry in data:
    print(f"{j} - Recognized Name: {entry[0]}, Confidence Score: {entry[1]}")
    j+=1
table = Table(title="Image Recognition Results", show_header=True)
table.add_column("Image", style="cyan", justify="center")
table.add_column("Recognized Name", style="cyan", justify="center")
table.add_column("Confidence Score", style="cyan", justify="center")

# Save images to temporary files
temp_image_files = []
for i, img_array in enumerate(images):
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_image_file.name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    temp_image_files.append(temp_image_file.name)

# Add data to the table
for i, entry in enumerate(data):
    table.add_row(f"image {i}",entry[0], f"{entry[1]}%")

# Convert the table to a Pandas DataFrame
print(table)