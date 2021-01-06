from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import cv2
import re
import os
import sys
import pygame
import time
import random

pygame.mixer.init()

def create_soundz(name):
    sound = pygame.mixer.Sound(name)
    sound.set_volume(0.27)
    return sound

pygame.mixer.set_num_channels(6)

# assign channels
one = pygame.mixer.Channel(0)
two = pygame.mixer.Channel(1)
three = pygame.mixer.Channel(2)
four = pygame.mixer.Channel(3)


soundA = create_soundz("MusicBoxA.wav")
soundB = create_soundz("MusicBoxB.wav")
soundC = create_soundz("MusicBoxC.wav")
soundD = create_soundz("MusicBoxD.wav")


# the TFLite converted to be used with edgetpu
modelPath = sys.argv[1]

# The path to labels.txt that was downloaded with your model
labelPath = sys.argv[2]

# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

# This function takes in a PIL Image from any source or path you choose
def classifyImage(image, engine):
    # Load and format your image for use with TM2 model
    # image is reformated to a square to match training
    image.resize((224, 224))

    # Classify and ouptut inference

   
    classifications = engine.classify_with_image(image, threshold =0.93, top_k=1)
    # classifications = engine.classify_with_image(image, threshold =0.71, top_k=1)
    # classifications = engine.classify_with_image(image, top_k=1)
    return classifications

def main():
    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)

    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Format the image into a PIL Image so its compatable with Edge TPU
        cv2_im = frame
        pil_im = Image.fromarray(cv2_im)

        # Resize and flip image so its a square and matches training
        pil_im.resize((224, 224))
        pil_im.transpose(Image.FLIP_LEFT_RIGHT)

        # Classify and display image
        results = classifyImage(pil_im, engine)
        cv2.imshow('frame', cv2_im)


        if results:
            print( labels[results[0][0]])

        # if results: print('Classification:', labels[results[0][0]], 'score:', str(results[0][1]))

         
               

      

            if ( labels[results[0][0]] == "Class 1" ):
                print ("one")
                if one.get_busy():
                   print("one busy signal")
                   time.sleep(0.17)
                   # adjust to 0.13
                else:
                   one.play(soundA)
                   r1 = random.randint(0,3)/10
                   #time.sleep(0.13)
                   time.sleep(r1)

            elif ( labels[results[0][0]] == "Class 2" ):
                print ("two")
                if two.get_busy():
                   print("two busy signal")
                   time.sleep(0.17)
                        # adjust to 0.13
                else:
                   two.play(soundB)
                   r2 = random.randint(1,4)/10
                   #time.sleep(0.13)
                   time.sleep(r2)

            elif ( labels[results[0][0]] == "Class 3" ):
                print ("three")
                if three.get_busy():
                   print("three busy signal")
                   time.sleep(0.17)
                        # adjust to 0.13
                else:
                   three.play(soundC)
                   r3 = random.randint(0,3)/10
                   # time.sleep(0.13)
                   time.sleep(r3)

            elif ( labels[results[0][0]] == "Class 4" ):
                print ("four")
                if four.get_busy():
                   print("four busy signal")
                   time.sleep(0.17)
                        # adjust to 0.13
                else:
                   four.play(soundD)
                   r4 = random.randint(1,4)/10
                   #time.sleep(0.13)
                   time.sleep(r4)

                       


            else:
                soundA.stop()
                soundB.stop()
                soundC.stop()
                soundD.stop()
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
