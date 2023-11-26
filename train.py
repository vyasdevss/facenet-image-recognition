
from os import listdir
from detect import *
from architecture import *
from random import choice
import os
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# load images and extract faces for all images in a directory
def load_faces(directory):
 faces = list()
 # enumerate files
 i = 0
 for filename in listdir(directory):
 # path
    if i>10:
      break
    path = directory + filename
 # get face
    try:  
     face = extract_face(path)
     i+=1
    except:
      continue
   
    if len(face) == 0:
      continue
 # store
    faces.append(face)
 return faces

def load_dataset(directory):
 X, y = list(), list()
 # enumerate folders, on per class
 i = 0
 for subdir in listdir(directory):
 # path
    path = directory + subdir + '/'
 # skip any files that might be in the dir
    if not os.path.isdir(path):
        continue
 # load all faces in the subdirectory
    faces = load_faces(path)
 # create labels
    labels = [subdir for _ in range(len(faces))]
 # summarize progress
    print('>loaded %d examples for class: %s' % (len(faces), subdir))
 # store
    X.extend(faces)
    y.extend(labels)

    try:
      data = np.load('fornow3.npz')
      xa = data['a'].tolist()
      xb = data['b'].tolist()
      xa.extend(X)
      xb.extend(y)
      np.savez_compressed('fornow3.npz', a = xa,b = xb)
      X.clear()
      y.clear()    
      print("done-{i}")
      i+=1
    except:
      print("nothing")
 
 trainX, valX, trainy, valy = train_test_split(xa, xb, test_size=0.1, random_state=42)
 return trainX, valX, trainy,valy


# load train dataset
trainX, valX, trainy, valy = load_dataset('Dataset/')



#saving dataset
np.savez_compressed('dataset.npz', trainX, trainy, valX,valy)
print(trainX.shape, trainy.shape)
trainX, valX, trainy, valy = load_dataset('Dataset/')
np.savez_compressed('dataset2.npz', trainX = trainX, trainy = trainy, valX = valX,valy = valy)
data = np.load('dataset2.npz')
trainX, trainy, testX, testy = data['trainX'], data['trainy'], data['valX'], data['valy']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

#loading Facenet
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
# load the facenet model
model = face_encoder

print('Loaded Model')
#creating embedings and saving them
newTrainX = list()
for face_pixels in trainX:
 embedding = get_embedding(model, face_pixels)
 newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
 embedding = get_embedding(model, face_pixels)
 newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)


data = load('embeddings.npz')
testX_faces = data['newTestX']

# load face embeddings
trainX, trainy, testX, testy = data['newTrainX'], data['trainy'], data['newTestX'], data['testy']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

unknown_labels = set(testy) - set(out_encoder.classes_)
if unknown_labels:
    print(f"Found unknown labels in test set: {unknown_labels}")
    # Add unknown labels to the encoder's classes
    out_encoder.classes_ = np.concatenate((out_encoder.classes_, list(unknown_labels)))

trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit SVM classifier model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# predicting scores
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
