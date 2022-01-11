import cv2

trained_face_data=cv2.CascadeClassifier("face-detector\\haarcascade_frontalface_default.xml")

img= cv2.imread("face-detector\\kevin.jpg")


grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceCoordinates=trained_face_data.detectMultiScale(grayscaled_img)

print(faceCoordinates)

for (x,y,w,h) in faceCoordinates:
    cv2.rectangle(img, (x , y) , (x+w,y+h) , (0,255,0),2)

cv2.imshow("image face detector :",img)
cv2.waitKey()
