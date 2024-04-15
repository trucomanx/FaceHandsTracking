#!/usr/bin/python
# pip3 install mediapipe
# pip3 show cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector




cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
facedetector = FaceDetector()



while True:
    success, img =cap.read();
    
    hands, img = detector.findHands(img);
    ## hands: list of dict
    ## dict: {  'lmList': np.array(21,3), 
    ##          'bbox': (502, 245, 207, 262), 
    ##          'center': (605, 376), 
    ##          'type': 'Right' }
    
    if hands:
        hand = hands[0];
        #print("\nhand:\n"hand)
    
    img, bboxs = facedetector.findFaces(img,draw=True);
    ## bboxs: list of dict
    ## dict: {  'id': 0, 
    ##          'bbox': (162, 136, 199, 199), 
    ##          'score': [0.9449860453605652], 
    ##          'center': (261, 235) }
    
    if bboxs:
        bbox = bboxs[0];
        print("\nFace:\n",bbox)
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
 
