import cv2


# Create our body classifier
face_cascade = cv2.CascadeClassifier('C:/Users/elven/Downloads/P106SecurityDetection/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/elven/Downloads/P106SecurityDetection/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grayimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    faces = face_cascade.detectMultiScale(grayimg,1.1,5)
    
    # Extract bounding boxes for any bodies identified
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('walkers',frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
