import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
video_capture=cv2.VideoCapture(0)


jobs_image=face_recognition.load_image_file('photos/jobs.jpg')
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file('photos/tata.jpg')
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

elon_musk = face_recognition.load_image_file('photos/elon.jpg')
elon_musk_encoding = face_recognition.face_encodings(elon_musk)[0]

known_face_encoding=[
    jobs_encoding,
    ratan_tata_encoding,
    elon_musk_encoding
]

known_faces_names=[
    "jobs",
    "ratan tata",
    "elon"
]

students=known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date +'.csv','w+',newline ='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]

                face_names.append(name)
                if name in known_faces_names:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendance system",frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

