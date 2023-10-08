import os
import sys

import cv2
import numpy as np

#
#  Instructions
#
# The images file name should start with "coins", followed by the number of coins in the image and the file extension.
# e.g.: "coins50.jpg".
#
# Other than that, there is no special configuration needed to run the script.
#
# Input 1: Enter the path to an image or to a directory of images. It can be a relative or an absolute path.
# Input 2: Option to enter "y" to choose Otsu method for binarization, instead of fixed threshold.
# Input 3: Option to enter "y" to see extra logs, images and masks.
#

# Ask for inputs.
path = input("Enter the path to an image or to a directory of images: ")
shouldUseOtsuInput = input(
    "Use Otsu method for binarization, instead of fixed threshold? (y/n) [default=n]: ")
isTestModeInput = input("Enable Test Mode? (y/n) [default=n]: ")

# Check if file is file or directory
if os.path.isfile(path):
    isFile = True
elif os.path.isdir(path):
    isFile = False
else:
    sys.exit(f"File/Directory not found in path: {path}")

# Wether to use Otsu method for binarization or fixed threshold (default)
shouldUseOtsu = shouldUseOtsuInput == "y" or shouldUseOtsuInput == "Y"

# Test Mode will show images, masks and extra logs
isTestMode = isTestModeInput == "y" or isTestModeInput == "Y"


# Function to predict the number of coins in an image
def predictCoinsCount(imgPath: str) -> int:
    img = cv2.imread(imgPath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if isTestMode:
        cv2.imshow("original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if isTestMode:
        cv2.imshow("grayscale", imgGray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Binarization
    if shouldUseOtsu:
        (t, imgBin) = cv2.threshold(255 - imgGray, 0, 255, cv2.THRESH_OTSU)
    else:
        (t, imgBin) = cv2.threshold(imgGray, 225, 255, cv2.THRESH_BINARY_INV)

    if isTestMode:
        cv2.imshow("binary", imgBin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    imgProcessed = imgBin

    # Closure
    # Intended to close the gaps inside the coins
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgProcessed = cv2.morphologyEx(imgProcessed, cv2.MORPH_CLOSE, strel)

    if isTestMode:
        cv2.imshow("binary closed", imgProcessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Erusion
    # Intended to "repel coins from each other"
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    imgProcessed = cv2.erode(imgProcessed, strel)

    if isTestMode:
        cv2.imshow("binary eroded", imgProcessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Opening
    # Intended to "break the coins away from each other"
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    imgProcessed = cv2.morphologyEx(imgProcessed, cv2.MORPH_OPEN, strel)

    if isTestMode:
        cv2.imshow("binary open", imgProcessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #
    (numLabels, labels, boxes, centroids) = cv2.connectedComponentsWithStats(imgProcessed)

    if isTestMode:
        for i in range(1, numLabels):
            x, y, w, h, area = boxes[i]
            color = list(np.random.random(size=3) * 256 / 2)
            center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
            radius = int(np.sqrt(area / np.pi))
            cv2.circle(img, (center_x, center_y), radius, color, 2)
        cv2.imshow("boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return numLabels - 1


def getBoolEmoji(value: bool) -> str:
    if value:
        return "✅"
    else:
        return "❌"


if isFile:
    # If the path points to a file, make prediction, and show results
    fileName = os.path.basename(path)
    label = int(fileName.replace("coins", "").split(".")[0])
    prediction = predictCoinsCount(path)
    print("\nLabel: Number of coins taken from file name")
    print("Prediction: Number of coins the algorithm found\n")
    print("   Filename, Label, Prediction")
    print("-------------------------------")
    print(f"{getBoolEmoji(prediction == label)} {fileName}, {label}, {prediction}")
else:
    # If the path points to a directory, make predictions, and show results
    fileNames = np.array(os.listdir(path))
    labels = np.array([int(fileName.replace("coins", "").split(".")[0]) for fileName in fileNames])
    predictions = []

    for fileName in fileNames:
        filePath = os.path.join(path, fileName)
        prediction = predictCoinsCount(filePath)
        predictions.append(prediction)
    predictions = np.array(predictions)

    results = np.column_stack((fileNames, labels, predictions))
    print("\nLabel: Number of coins taken from file name")
    print("Prediction: Number of coins the algorithm found\n")
    print("   Filename, Label, Prediction")
    print("-------------------------------")
    for result in results:
        (fileName, label, prediction) = result
        print(f"{getBoolEmoji(prediction == label)} {fileName}, {label}, {prediction}")

    nCorrectPredictions = np.sum(predictions == labels)
    percCorrectPredictions = nCorrectPredictions / len(labels) * 100
    print(f"\nCorrect Predictions: {nCorrectPredictions}/{len(labels)} ({percCorrectPredictions:.2f}%)")

    nTotalLabels = np.sum(labels)
    nTotalPredictions = np.sum(predictions)
    percCount = nTotalPredictions / nTotalLabels * 100
    print(f"\nTotal Count: {nTotalPredictions}/{nTotalLabels} ({percCount:.2f}%)")
