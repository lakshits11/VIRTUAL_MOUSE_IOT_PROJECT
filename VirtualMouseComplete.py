import cv2
import autopy
import time
import numpy
import HandTrackingModule as htm

frameVertex = 100  # Defining vertices of Frame

# Defining width and height in pixels to be captured by camera
cameraHeight = 720
cameraWidth = 1280

smoothness = 7

previousTime = 0
# Previous Location of coordinate X and Y
prevLocationX, prevLocationY = (0, 0)
# Current Location of coordinte X and Y
currentLocationX, currentLocationY = 0, 0

# Capturing the video from
captureVideo = cv2.VideoCapture(0)

# Setting camera height and width
captureVideo.set(4, cameraHeight)  # id for setting height is 4
captureVideo.set(3, cameraWidth)  # id for setting width is 3

# We need to detect the hand. Here in handDetector module, we have to detect only one hand.
# Therefore, we put maxHands=1 in detect method.
detector = htm.handDetector(maxHands=1)

# Getting the width and height of the screen.
screenWidth, screenHeight = autopy.screen.size()
# print("Width: ", screenWidth, "Height: ", screenHeight)

while True:

    # Find hand points to be tracked
    success, img = captureVideo.read()
    img = detector.findHands(img)
    # We are passing the image in detector.findPosition method and it will give the bounding box of hands in image.
    lmList, boundingBox = detector.findPosition(img)

    # Get the tip of the index and middle finger
    if len(lmList) != 0:
        # xIndexFinger and yIndexFinger are the coordinates of the index finger.
        xIndexFinger, yIndexFinger = lmList[8][1:]
        # xMiddleFinger and yMiddleFinger are the coordinates of the middle finger.
        xMiddleFinger, yMiddleFinger = lmList[12][1:]
        # print(xIndexFinger, yIndexFinger, xMiddleFinger, yMiddleFinger)

        # Finding information about which fingers are up.
        fingers = detector.fingersUp()
        # print(fingers)

        # We are creating a rectangular region in which we want to detect the hand.
        # That is if our hand lies in that rectangle region, then only we want it to detect.
        cv2.rectangle(
            img,
            (frameVertex, frameVertex),
            (cameraWidth - frameVertex, cameraHeight - frameVertex),
            (255, 0, 255),
            2,
        )

        # Since the mouse moves by detecting the movement of our index finger
        # So here we are checking if only index finger is up, then only move the mouse.
        # Here we are also checking if the middle finger is not up, then we are moving the mouse.
        # Whenever middle finger will also be up with index finger, then we will use that for click functionality.
        if fingers[1] == 1 and fingers[2] == 0:

            # Convert coordinates
            # This is because the webcam will give value of 1280 by 720 pixels while the screen is 1920 by 1080.
            # So, we need to convert the coordinates to get correct coordinates.
            convertedXcoordinate = numpy.interp(
                xIndexFinger, (frameVertex, cameraWidth - frameVertex), (0, screenWidth)
            )
            convertedYcoordinate = numpy.interp(
                yIndexFinger,
                (frameVertex, cameraHeight - frameVertex),
                (0, screenHeight),
            )

            # We need to smoothen the value to reduce flickering and jitterness.
            currentLocationX = (
                -(prevLocationX - convertedXcoordinate) / smoothness + prevLocationX
            )
            currentLocationY = (
                -(prevLocationY - convertedYcoordinate) / smoothness + prevLocationY
            )

            # Move Mouse by tracking hand position
            # We did screenWidth-currentLocationX because we need to move the mouse in direction of hand.
            # If we had directly given params as currentLocationX and currentLocationY, then the mouse would have
            # moved in opposite direction of the movement of hand.
            autopy.mouse.move(screenWidth - currentLocationX, currentLocationY)

            # Whenever we are in moving mode, it will show a circle at the tip of index finger to aid us.
            cv2.circle(img, (xIndexFinger, yIndexFinger), 15, (255, 0, 255), cv2.FILLED)
            # Updating previous location of mouse
            prevLocationX, prevLocationY = currentLocationX, currentLocationY

        # Now if both index and middle finger are up, then we are clicking the mouse.
        if fingers[1] == 1 and fingers[2] == 1:
            # Finding the distance between the index finger and middle finger.
            # foundDistance is the distance between the index finger and middle finger.
            foundDistance, img, lineInfo = detector.findDistance(8, 12, img)
            print(foundDistance)
            # On observing, we found that if fingers are near enough, the foundDistance was aroung 39-40.
            # So we will use that info to click the mouse.
            # Click the mouse if distance is short
            if foundDistance < 39:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Frame Rate
    currentTime = time.time()
    framesPerSecond = -1 / (previousTime - currentTime)
    previousTime = currentTime
    # print(framesPerSecond)

    # Displaying the fps on the screen
    # (80,70) is the position of the text
    cv2.putText(
        img,
        str(int(framesPerSecond)),  # Text to be put
        (80, 70),
        cv2.FONT_HERSHEY_COMPLEX,  # Font to be used
        3,  # Font scale
        (255, 105, 18),
        3,  # Thickness of the text
    )

    cv2.imshow("Img", img)
    cv2.waitKey(1)
