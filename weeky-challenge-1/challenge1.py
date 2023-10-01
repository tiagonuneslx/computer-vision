import os
import sys

import cv2
import numpy as np

#
#  Instructions
#
# There is no special configuration needed to run the script.
#
# Input 1: Enter the path to an image or to a directory of images. It can be a relative or an absolute path.
# Input 2: Option to enter "y" to see extra logs, images and masks.
#


# Ask for inputs.
path = input("Enter the path to an image or to a directory of images: ")
isTestModeInput = input("Enable Test Mode? (y/n) [default=n]: ")

# Check if file is file or directory
if os.path.isfile(path):
    isFile = True
elif os.path.isdir(path):
    isFile = False
else:
    sys.exit(f"File/Directory not found in path: {path}")

# Test Mode will show images, masks and extra logs
isTestMode = isTestModeInput == "y" or isTestModeInput == "Y"


# Classifier function to predict if the image is a beach
def predictIfBeach(imgPath: str) -> bool:
    # Read image
    img = cv2.imread(imgPath)
    height, width = img.shape[0], img.shape[1]

    if isTestMode:
        cv2.imshow(imgPath, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Is bright enough (50% of pixels above 50% gray)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (t, maskGray) = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
    isBright = np.sum(maskGray > 0) / np.size(maskGray) > 0.5

    if isTestMode:
        print("Is bright enough (50% of pixels above 50% gray):", isBright)
        cv2.imshow("brightness mask", cv2.bitwise_and(img, img, mask=maskGray))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Is top half blue enough (40% of top half pixels blue)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskBlue = cv2.inRange(imgHsv, (90, 0, 70), (128, 255, 255))
    maskBlueTop = np.array(maskBlue[:int(height / 2), :])
    isTopBlue = np.sum(maskBlueTop > 0) / np.size(maskBlueTop) > 0.4

    if isTestMode:
        print("Is top blue enough (40% of top half pixels blue):", isTopBlue)
        maskBlue[int(height / 2):, :] = 0
        cv2.imshow("blue mask", cv2.bitwise_and(img, img, mask=maskBlue))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Is bottom half sandy enough (20% of bottom half pixels sand)
    maskSand = cv2.inRange(imgHsv, (0, 0, 20), (30, 180, 255))
    maskSandBottom = np.array(maskSand[int(height / 2):, :])
    isBottomSand = np.sum(maskSandBottom > 0) / np.size(maskSandBottom) > 0.2

    if isTestMode:
        print("Is bottom sandy enough (20% of bottom half pixels sand):", isBottomSand)
        # Show sand mask
        maskSand[:int(height / 2), :] = 0
        cv2.imshow("sand mask", cv2.bitwise_and(img, img, mask=maskSand))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return isBright and isTopBlue and isBottomSand


def getBoolEmoji(value: bool) -> str:
    if value:
        return "✅"
    else:
        return "❌"


if isFile:
    # If the path points to a file, predict if the image is a beach, and show results
    fileName = os.path.basename(path)
    label = fileName.startswith("beach")
    prediction = predictIfBeach(path)
    print("\nLabel: True if is beach, False otherwise\n")
    print("   Filename, Label, Prediction")
    print("-------------------------------")
    print(f"{getBoolEmoji(prediction == label)} {fileName}, {label}, {prediction}")
else:
    # If the path points to a directory, predict if each image is a beach, and show results
    fileNames = np.array(os.listdir(path))
    labels = np.array([True if fileName.startswith("beach") else False for fileName in fileNames])
    predictions = []

    for fileName in fileNames:
        filePath = os.path.join(path, fileName)
        prediction = predictIfBeach(filePath)
        predictions.append(prediction)
    predictions = np.array(predictions)

    results = np.column_stack((fileNames, labels, predictions))
    print("\nLabel: True if is beach, False otherwise\n")
    print("   Filename, Label, Prediction")
    print("-------------------------------")
    for result in results:
        (fileName, label, prediction) = result
        print(f"{getBoolEmoji(prediction == label)} {fileName}, {label}, {prediction}")

    nCorrectPredictions = np.sum(predictions == labels)
    percCorrectPredictions = nCorrectPredictions / len(labels) * 100
    print(f"\nCorrect Predictions: {nCorrectPredictions}/{len(labels)} ({percCorrectPredictions:.2f}%)")
