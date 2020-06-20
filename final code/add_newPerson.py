import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data=[]
dataset_path='./data/'
file_name=input("enter your name:")
skip=0 
while True:
 	rel,frame=cap.read()
 	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 	if rel==False:
 		continue
 	faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
 	faces=sorted(faces,key=lambda f:f[2]*f[3])
 	
 	
 	for face in faces[-1:]:
 		x, y, w, h = face
 		offset=10
 		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
 		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
 		face_section=cv2.resize(face_section,(100,100))
 		face_data.append(face_section)
 		
 		cv2.imshow('FaceSection',face_section)
 	cv2.imshow('the video of u',frame)	
 		
 	
 	num=cv2.waitKey(1)&0xFF
 	if num==ord('q'):
 		break 
cap.release()
cv2.destroyAllWindows()		
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+".npy",face_data)