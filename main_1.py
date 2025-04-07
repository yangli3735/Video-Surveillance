# -*- coding: utf-8 -*-

import datetime
import imutils
import warnings
import time
import cv2
import numpy as np
import threading
import mimetypes
from imutils.object_detection import non_max_suppression
import smtplib
from email.message import EmailMessage


def send_email(to_email, frame):
    now = datetime.datetime.now()
    port = 465
    app_password = 'ztfobdckrbodecon'  # Replace securely or load from env variables
    msg = EmailMessage()
    msg['Subject'] = 'Possible Intrusion Detected at ' + now.strftime("%m/%d/%Y, %H:%M:%S")
    msg['From'] = 'yxl3735@case.edu'
    msg['To'] = to_email

    image_path = './Common/email.png'
    cv2.imwrite(image_path, frame)

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type, mime_subtype = mime_type.split('/')

    with open(image_path, 'rb') as fp:
        img_data = fp.read()
        msg.add_attachment(img_data, maintype=mime_type, subtype=mime_subtype, filename='alert.png')

    with smtplib.SMTP_SSL("smtp.gmail.com", port) as server:
        server.login("yxl3735@case.edu", app_password)
        server.send_message(msg)


# Parameters
min_area = 5000
min_motion_frames = 5
thresh_width = 100
delta_thresh = 5
show_video = True

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
email_alert_enabled = False  # Set True to enable email alert

avg = None
motion_counter = 0
warnings.filterwarnings("ignore")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not camera.isOpened():
    print("[Error] Can't open camera!")

print("[Info] Camera warming up...")
time.sleep(2)

while True:
    ret, frame = camera.read()
    time.sleep(0.5)

    if ret:
        timestamp = datetime.datetime.now()
        text = "Safe"

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        boxes_face = cascade_face.detectMultiScale(gray, 1.1, 5)
        face_area = 0

        for (xf, yf, wf, hf) in boxes_face:
            face_area = wf * hf
            cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)

        if avg is None:
            print("[Info] Initializing background model...")
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frame_delta, delta_thresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 1)
            if (xB - xA) > thresh_width:
                text = "Suspicious"

        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if w > thresh_width and face_area > 0:
                face_area = 0
                text = "Suspicious"

        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, f"Room Status: {text}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)

        if text == "Suspicious":
            motion_counter += 1
            if motion_counter >= min_motion_frames:
                print("Alert!")
                _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                data = img_encoded.tobytes()
                dec_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                
                if email_alert_enabled:
                    threading.Thread(target=send_email, args=('rxw496@case.edu', dec_img)).start()
                    print("Sending alert email")

                cv2.imshow('Alert Frame', dec_img)
                motion_counter = 0

        else:
            motion_counter = 0

        if show_video:
            cv2.imshow("Security Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
