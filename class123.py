import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plp
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import PIL
from PIL import Image
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

cap = cv2.VideoCapture(0)
x = True
while x == True:
   try:
    ret, frame= cap.open()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #draw the box
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #roi 
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    # cv2 to pil format
    image_pil = Image.fromarray(roi)
    # convert every pixel ( 'L' ) to gray scale 
    gray_scale = image_pil.convert('L')
    resize_image = gray_scale.resize((28,28), Image.ANTIALIAS)
    image_inverted = PIL.ImageOps.invert(resize_image)

    pixel_filter = 20
    min_pixel = np.percentile(image_inverted, pixel_filter)

    cliped_image = np.clip(image_inverted-min_pixel,0,255)
    max_pixel = np.max(image_inverted)

    scaled_image = np.asarray(cliped_image)/max_pixel
    test_value = np.array(scaled_image).reshape(1,784)
    
    predict = clf.predict(test_value)

    #display the result 
    cv2.imshow('Output', gray)
    if(cv2.waitKey(1)  & 0xFF == ord('q')):
        x = False
      
   except Exception as e:
    pass
    
