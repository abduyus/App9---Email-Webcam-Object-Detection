import streamlit as st
import cv2
import time
import glob
import os
from threading import Thread
from modules.emailing import send_email

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []
count = 1
email_sent = False  # Track whether an email has been sent

def clean_folder():
    print("Clean Folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("Clean Folder function ended")

st.title("Motion Detection App")

st.write("Live Webcam Feed:")

while True:
    status = 0
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_frame_gau
    
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    st.image(frame, channels="BGR", use_column_width=True, caption="Live Webcam Feed")

    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if frame.any() and not email_sent:
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:]
    st.write("Status List:", status_list)

    if status_list[0] and status_list[1] == 0 and not email_sent:
        email_thread = Thread(target=send_email, args=(image_with_object,))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True
        
        email_thread.start()
        email_sent = True
        st.write("Email sent!")

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

clean_thread.start()
video.release()
