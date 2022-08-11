import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import os
from numpy import asarray
import csv
from PIL import Image
import math
import pandas as pd
from sklearn import preprocessing
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def CreateScores(NumberofElements):
    AllAnswers = []
    for x in range(NumberofElements):
        Low = 1; High = 9; Difference = High - Low
        AnsweredQuestion = random.randrange(Low, High)
        AllAnswers.append(AnsweredQuestion)
    return  AllAnswers, Difference

def multiply_pixel(n):
    m = n * 30
    return m

def create_series(TheList):
    Theseries = pd.Series(TheList)
    return Theseries


def sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return (sum)

def Create_RGB_tuples(List_In):
    res = [(val,val,val) for val in List_In]
    return res

def merge_Lists(ListA, ListB):
    ListA[:len(ListB)] = ListB
    return ListA

def Standardize(InDataframe):
    # Get column names first
    Dataframe = InDataframe.copy()
    col_names = Dataframe.columns
    features = Dataframe[col_names]
    Standardized = preprocessing.StandardScaler()
    scaler = Standardized.fit(features.values)
    features = scaler.transform(features.values)
    Dataframe[col_names] = features
    return Dataframe

def MinMaxScaling(train, test):
    """
    Pre-processes the given dataframe by minmaxscaling the continuous features (fit-transforming the training data and
    transforming the test data)
    """

    mms = MinMaxScaler()
    trainX = mms.fit_transform(train.columns)
    testX = mms.transform(test.columns)
    return (trainX, testX)


def Questionnaire_Image(width=5, height=5, Item_vals=0):
    # Apply the function higher_power to all elements in the Item_vals list.
    empowered_list = list(map(multiply_pixel, Item_vals))
    # Create a list as long as the number of pixels in the new image.
    blank_image_list = [255 for _ in range(width * height)]
    # Merge the list (of questions) with that of the list for image pixels
    Q_Image_list = merge_Lists(blank_image_list,empowered_list)
    Tuples = Create_RGB_tuples(Q_Image_list)
    Outputimagesize = (width, height)
    dst_image = Image.new('RGB', Outputimagesize)
    dst_image.putdata(Tuples)
    return dst_image

def QuestionnairetoFile(QuestionnaireArray):
    # Write the feature lists to file.
    with open(Folder + "/Synthetic_Answers.txt", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(QuestionnaireArray)


def buildModel(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    return model

def build(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()

    inputShape = (height, width, depth)
    chanDim = -1

    # if we are using "channels first", update the input shape and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # CONV => RELU => POOL
    model.add(SeparableConv2D(32, (3, 3), padding="same",
                              input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

Folder = 'Images/'
Elements = 100
QuestionnaireArrays, Bins = CreateScores(Elements)
QuestionnaireDF = pd.DataFrame()

QuestionnaireDF = pd.DataFrame({'Label':QuestionnaireArrays})
# QuestionnaireDF["Label"] = Binned_List

# When there are multiple questions the image size depends on how many questions there are.
path = "Basic_Image"
width = height = int(math.ceil(math.sqrt(len([QuestionnaireDF["Label"][0]]))))

# if the label output directory does not exist, create it
if not os.path.exists(path):
    print("[INFO] 'creating {}' directory".format(path))
    os.makedirs(path)

ImageArray = []
width = 48
height = 48
imageSize = (width, height)
for i in range(len(QuestionnaireDF["Label"])):
    # Create a 1px x 1px image based on the Label value.
    LabelValue = [QuestionnaireDF.iloc[i]["Label"]]
    image = Questionnaire_Image(1, 1, Item_vals=LabelValue)
    image = cv2.resize(asarray(image), imageSize)
    ImageArray.append(image)
    image = Image.fromarray(image)
    filename = str(LabelValue) + "_image_" + str(i)
    image.save(path + "/{file_name}.png".format(file_name=filename))

print("[INFO] processing data...")
QuestionnaireData = np.array(QuestionnaireDF["Label"])
TheList = []
[TheList.append([i]) for i in QuestionnaireData]
QuestionnaireDF["Label"] = TheList

QuestionnaireData = np.array(QuestionnaireDF["Label"])

ImageArray = np.array(ImageArray)
split = train_test_split(ImageArray, QuestionnaireData,  test_size=0.25)
(train_images, test_images, train_labels, test_labels) = split


trainCategorical = MultiLabelBinarizer().fit_transform(train_labels)
testCategorical = MultiLabelBinarizer().fit_transform(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = build(width=width, height=height, depth=3,classes=Bins)
model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

model.fit(train_images, trainCategorical, epochs=10)

predIdxs = model.predict(x=test_images)

predIdxs = np.argmax(predIdxs, axis=1)

test_labels = np.hstack(test_labels)
Accuracy = accuracy_score(test_labels,predIdxs )
Precision = precision_score(test_labels,predIdxs , average="macro")
Recall = recall_score(test_labels,predIdxs , average="macro")
Fscore = f1_score(test_labels,predIdxs , average="macro")

print(Accuracy)
print(Precision)
print(Recall)
print(Fscore)