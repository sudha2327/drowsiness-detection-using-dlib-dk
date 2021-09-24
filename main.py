from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
frequency=2500
duration=1000

def eyeAspectration(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear



count=0
eyeThresh=0.3
eyeFrames=48
blink=0

shapePredictor="shape_predictor_68_face_landmarks.dat"



detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(shapePredictor)
print("predictor successfully.....")

cam=cv2.VideoCapture(0)
print("camera has opened.....")
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("while loop formate.....")
while True:
    _,frame=cam.read()
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print("converted image to gray successfully......")
    rects=detector(gray,0)

    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]

        leftEar=eyeAspectration(leftEye)
        rightEar=eyeAspectration(rightEye)

        ear=(leftEar+rightEar)/ 2.0

        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,0,255),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,0,255),1)

        if ear < eyeThresh :
            count+=1

            if count >=eyeFrames:
                cv2.putText(frame,"DROWSINESS DETECTED",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                winsound.Beep(frequency,duration)
                blink=0
            elif  count >=1 & count < eyeFrames:
                blink+=1

        else:
            count=0
        cv2.putText(frame,"blink counter{}".format(int(blink/10)),(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)



    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF

    if key==ord("q"):

        break

cam.release()
cv2.destroyAllWindows()



