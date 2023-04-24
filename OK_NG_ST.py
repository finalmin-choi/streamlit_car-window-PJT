# 필요한 라이브러리를 import
import streamlit as st
import ultralytics
ultralytics.checks()
import cv2
# import torch
import numpy as np
import os

# import sys
from PIL import Image
from ultralytics import YOLO
model = YOLO('best.pt')

import requests
from io import BytesIO

st.title("Order Checking in Drive Thru")

instructions = """
        주문 진행 유무 판단 후 팬 속도 조절
        """
st.write(instructions)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
  if labels == []:
    labels = {0: u'car', 1: u'closed', 2: u'opened'}
  #Define colors
  if colors == []:
    #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [(0, 255, 0),(0, 0, 255),(255, 0, 0)]

  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        box_label(image, box, label, color)
    else:
      color = colors[int(box[-1])]
      box_label(image, box, label, color)


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

import os
import streamlit as st

file = st.file_uploader('Select An Image')

if file:  # if user uploaded file
    img = Image.open(file)
    # img.save('/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/')

    st.header("주문 진행 유무 확인 이미지")
    resized_image = img.resize((640, 640))
    st.image(resized_image)

    resized_image.save("C:/Users/heuser/Desktop/window-project/streamlit_car-window-PJT/ST_test_image/deter_image/selection_image.jpg")
    st.success("The image has been saved to local!")

    image = cv2.imread("C:/Users/heuser/Desktop/window-project/streamlit_car-window-PJT/ST_test_image/deter_image/selection_image.jpg") 

    results = model.predict(image)

    plot_bboxes(image, results[0].boxes.data, score=False, conf=0.5)

    A = [0, 0]
    for j in range(len(results[0].boxes.data)):
      a = results[0].boxes.data[j]
      if a[5] == 0:
        A[0] = 1
      elif a[5] == 2:
        A[1] = 1

    if sum(A) == 2:
        print("현재 주문 진행 중... 팬 속도 감소!")
        text = 'Fan speed has decreased!'
        color = (255, 0, 0) # (B, G, R)
        save_img_path3 = 'C:/Users/heuser/Desktop/window-project/streamlit_car-window-PJT/ST_test_image/result_image/function.jpg'
        st.subheader('현재 주문 진행 중... 팬 속도 감소!')
    else:
        print('팬 정상 동작!')
        text = "Fan is operating normally!"
        color = (0, 0, 255) # (B, G, R)
        save_img_path3 = 'C:/Users/heuser/Desktop/window-project/streamlit_car-window-PJT/ST_test_image/result_image/normal.jpg'
        st.subheader('팬 정상 동작!')  


    # 텍스트 쓰기
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 3
    position = (50, 50) # 좌측 상단의 위치

    cv2.putText(image, text, position, font, font_scale, color, thickness)

    # 이미지 파일 저장
    cv2.imwrite(save_img_path3, image)

    st.image(image)