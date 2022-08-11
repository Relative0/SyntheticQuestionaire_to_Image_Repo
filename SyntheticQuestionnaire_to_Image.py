# Use locations of item along with their values to determine colors. I could then put them as unique colors at random locations.

# Maybe, for each subjects image each pixel location corresponds to a particular question and each color for each pixel corresponds
# to the answer.

# Or perhaps make some of the encoding/decoding outside the image e.g. each pixel location correpsonds to a prime number (so each location
# corresponds to a question) then each pixel color value corresponds to that prime number to the power of the answer. This will only
# work for images containing the number of questions less than the fifth root of 256^3 (for 1-5 answers). This will give a unique Godel number. Looks like
# I can get 27.85, so 27 questions out of this. Then making it a square image, I can do a 5x5 = 25 question image.
# - Seems that there ar problems with this, first as the 27th prime number is 103 and 103^5 is greater than 256^3, I had forgotten that
# there are gaps between prime numbers.
# I could instead just make the pixel locations the questions and make their colors not unique but all following a pattern of being
# one of 5 colors (one color for each of the answers). Or maybe make more colors by multiplying each of the 5 potential color values
# by pixel locations.
# We could also build a unique number by stacking each of the answers. So for example question 1,2,and 3 have a value of 3,5,4 respectively
# which would give the number 354 and as long as the questions only have values 0-9 we have a unique number. Now I could just give a color
# coding simply based on the answer or I could multiply the answer by the location and convert that in to RGB values. Perhaps I could
# also factor in the sum of scores on the set of questions as a sort of hash check for each image. That is, each of the images has a color
# coding decided in part by the overall sum of scores.

# Now to create a basic image from a column of questions with just 5 potential colors.

# for each column, need to translate that to a pixel value

import random
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import random
import uuid
import math
import csv
import os

from PIL import Image, ImageDraw

def higher_power(n):
    m = pow(n,3)
    return m

def Create_RGB_tuples(List_In):
    res = [(val,val,val) for val in List_In]
    return res

def merge_Lists(ListA, ListB):
    ListA[:len(ListB)] = ListB
    return ListA

def generate_random_image(width=128, height=128):
    rand_pixels = [random.randint(0, 255) for _ in range(width * height * 3)]
    rand_pixels_as_bytes = bytes(rand_pixels)
    text_and_filename = str(uuid.uuid4())
    random_image = Image.frombytes('RGB', (width, height), rand_pixels_as_bytes)
    draw_image = ImageDraw.Draw(random_image)
    draw_image.text(xy=(0, 0), text=text_and_filename, fill=(255, 255, 255))
    random_image.save(path+"/{file_name}.png".format(file_name=text_and_filename))

def Questionnaire_Image(width=5, height=5, Item_vals=0):
    # Apply the function higher_power to all elements in the Item_vals list.
    empowered_list = list(map(higher_power, Item_vals))
    # Create a list as long as the number of pixels in the new image.
    blank_image_list = [255 for _ in range(width * height)]
    # Merge the list (of questions) with that of the list for image pixels
    Q_Image_list = merge_Lists(blank_image_list,empowered_list)
    Tuples = Create_RGB_tuples(Q_Image_list)
    Outputimagesize = (width, height)
    dst_image = Image.new('RGB', Outputimagesize)
    dst_image.putdata(Tuples)  # Place pixels in the new image.
    # print(Tuples)
    return dst_image

def findweights():
    binarychoice = [0, 1]
    choice = random.choice(binarychoice)
    if choice == 0:
        tuple = (1,2)
    elif choice == 1:
        tuple = (2,1)
    else:
        print("Issues in findweights()")
    return tuple

def AnswerQuestions(numberofQuestions,DirichletProbs):
    test = np.random.choice([1, 2, 3, 4, 5], numberofQuestions, p=DirichletProbs)
    return test


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=20):
  dataframe = dataframe.copy()
  # labels = dataframe.pop('target') becomes the two columns we don't want in training:
  labels = dataframe.pop('The_Category')
#   print(dict(dataframe))
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  # print(ds)
  return ds

def write_greyscaleImage_toFile(subject, Image,width, height, ImagePath):
    # Write image to file
    Image_filename = "Image_" + str(subject)
    Image.save(ImagePath + "/{Image_file_name}.png".format(Image_file_name=Image_filename))

ListofVals= []
Questions = 8
Subjects = 20
QuestionsandProbabilities = []
AllSubjectsAnswers = []
for x in range(Subjects):
    a, b = findweights()
    # Change a and b multipliers to change graph thickness
    [DirichletProbabilities] = np.random.dirichlet((a, 2 * a, 2 * (a**2 + b**2), 2 * b, b), size=1).round(10)
    AnsweredQuestion = AnswerQuestions(Questions,DirichletProbabilities)
    AllSubjectsAnswers.append(AnsweredQuestion)
    # ListofVals.append(DirichletProbabilities.tolist())
    # QuestionsandProbabilities.append([AnsweredQuestion, DirichletProbabilities])
path = 'Synthetic_Questionnaire_Images'
# if the label output directory does not exist, create it
if not os.path.exists(path):
    print("[INFO] 'creating {}' directory".format(path))
    os.makedirs(path)
with open(path+"/Synthetic_Questionnaire_Answers.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(AllSubjectsAnswers)

width = height = int(math.ceil(math.sqrt(Questions)))

for Subject in range(len(AllSubjectsAnswers)):
    image = Questionnaire_Image(width, height, Item_vals=AllSubjectsAnswers[Subject])
    new_width = new_height = 300
    large_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    write_greyscaleImage_toFile(Subject, large_image, new_width, new_height, path)

    # image = Image.open("Images_5_Questions/" + filename + ".png")
    # print(list(image.getdata()))

