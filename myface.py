#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:18:54 2019

@author: yancykahn
"""

import os
from matplotlib import pyplot as plt
import cv2
import pyttsx3
from PIL import Image
import json
import numpy as np


# say hello
def SayHello(myname):
    audio = pyttsx3.init()
    audio.say("hello " + myname)
    audio.runAndWait()    

def SaySB(sbname):
    audio = pyttsx3.init()
    audio.say("sb " + sbname)
    audio.runAndWait()  
    
def SaySth(string):
    audio = pyttsx3.init()
    audio.say(string)
    audio.runAndWait()
    

#json 读取写入
def json_write(json_str):
    with open(r"FaceDataConfig/name.json","w") as f:
        json.dump(json_str, f)

def json_read():
    with open(r"FaceDataConfig/name.json","r") as f:
        json_str = json.load(fp=f)
        return json_str


facexml = r"/usr/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"

#获取特征图
def GetFeature():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库
    
    name2id=json_read()
    
    SaySth("please enter you name")
    face_id = input('\n enter user name:')
    name2id[str(len(name2id))]= face_id#添加映射
    json_write(name2id) #写入json
    
    if (os.path.exists('./FaceData/'+str(face_id))): # 判断是否存在目录
        pass
    else:
        os.mkdir('./FaceData/'+str(face_id)+'/')
    
    count = (len(os.listdir('./FaceData/'+face_id)))
    tcount = 0
    while(True):
        ret, frame = cap.read() # 读取一帧的图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰
    
        faces = face_cascade.detectMultiScale(gray, 
            scaleFactor = 1.15, 
            minNeighbors = 5, 
            minSize = (5, 5)
            ) # 检测人脸

        Area=0.0
        test_img=[]
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 用矩形圈出人脸
            if (w*h > Area and float(w + h) > 300.0):
                test_img=gray[y:y+h, x:x+w]
                Area = w*h
            
        plt.imshow(test_img)
        plt.show()
        
        if len(test_img) != 0:
            plt.imshow(test_img)
            plt.show()
            cv2.imwrite('./FaceData/'+str(face_id)+'/User.'+str(face_id)+'.'+str(count)+'.jpg', test_img) #保存最大特征图
            count = count + 1
            tcount = tcount + 1
            
        
        cv2.imshow('video', frame)
        print(tcount)
        if tcount == 100:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()


#获取特征和id
def GetFeatureAndLabels(path):
    Features = [f for f in os.listdir(path)]
    imgPaths = []
    FaceSamples = []
    ids = []
    names = []
    
    name2id = json_read()
    for feature in Features:
        test_path = os.path.join(path, feature)
        imgPaths.append([os.path.join(test_path, f) for f in os.listdir(test_path)])
        for imgpath in os.listdir(test_path):
            print(imgpath)
            PIL_img = Image.open(os.path.join(test_path,imgpath)).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            FaceSamples.append(img_numpy)
            names.append(feature)
            
            test_id = -1
            for i in range(len(name2id)):
                if name2id[str(i)] == feature:
                    test_id = i
                    break
            print(i)
            ids.append(int(test_id))
    
    #print(Features)
    #print(imgPaths) 
    return FaceSamples, ids, names
    
 
    
# 训练
def FaceTrain():
    print ("Train Face .... ")
    
    faces,ids,names=GetFeatureAndLabels('FaceData')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))   #开始训练
    print(recognizer)
    
    recognizer.write(r'FaceDataConfig/trainer.yml')

    print("{0} faces is trained".format(len(np.unique(ids))))




def FaceDetection():
    #获取当前存储的不知道的图片数
    if (os.path.exists('./FaceData/unknown/')):# 判断是否存在目录
        pass
    else:
        pass
        #os.mkdir('./FaceData/unknown')
        

    #count=len(os.listdir(r'./FaceData/unknown'))
    #print("unknown = ", count)
    
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库
    
    recoginzer = cv2.face.LBPHFaceRecognizer_create()
    recoginzer.read(r'FaceDataConfig/trainer.yml')  #加载分类器
    
    name2id=json_read()

    
    count=len(os.listdir(r'./FaceData/unknown'))
    print(count)
    name = set([])
    
    while(True):
        ret, frame = cap.read() # 读取一帧的图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰
    
        faces = face_cascade.detectMultiScale(gray, 
            scaleFactor = 1.15, 
            minNeighbors = 5, 
            minSize = (5, 5)
            ) # 检测人脸

        
        test_name = set([])
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 用矩形圈出人脸
            idnum, confidence = recoginzer.predict(gray[y:y+h, x:x+w])
            print ("id = ", idnum, "confidence = ", confidence)
            
            if confidence < 100:
                idnum = name2id[str(idnum)]
                confidence = "{0}%".format(round(100-confidence))
            else:
                idnum = "unknown"
                confidence = "{0}%".format(round(100-confidence))
            
           # if idnum == "unknown":
                #cv2.imwrite('./FaceData/unknown/User.unknown.'+str(count)+'.jpg', gray[y:y+h, x:x+w]) #保存未知图
            #    count+=1
             #   plt.imshow(gray[y:y+h, x:x+w])
              #  plt.show()
                
            cv2.putText(frame, str(idnum), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
            cv2.putText(frame, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
            
            test_name.add(idnum)
            
        

        
        print("Number of face: ", len(faces))
        cv2.imshow('video', frame)
        
        
        for tname in test_name:
            if tname not in name:
                if tname == "unknown":
                    SaySB(tname)
                else:
                    SayHello(tname)
            name.add(tname)
        
        if count == 100:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()
    

facexml = r"/usr/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"


        
def main():
    SaySth("Prepar Envoriment")
    os.system("pip3 install -U requirement.txt")
    SaySth("Welcome to my Face")
    
    
    while (True):
        SaySth("please chose a mode")
        SaySth("0 start register")
        SaySth("1 adminstrator mode")
        SaySth("q exit System")
        
        order = input("\n please input a order(0(register),1(adminstrator),q(exit))")
        if order == "q":
            SaySth("Thanks for you use, GoodBye")
            break
        elif order == "0":    
            if (os.path.exists('./FaceDataConfig/trainer.yml')):# 判断是否存在目录
                SaySth("face detection")
                FaceDetection()
            else:
                SaySth("Training Documnet is not found, please contact adminstrator")
                continue
        elif order == "1":
            SaySth("Welcome to Managment System")
            SaySth("Can I Help You")
            SaySth("t is Training")
            SaySth("g is Get Face Feature")
            mod = input("\n please input a mod, t(training), g(get face)")
            if mod == "t":
                if (os.path.exists('./FaceData') & (int)(len([os.listdir('./FaceData')])) >= 1):# 判断是否存在目录
                    SaySth("training")
                    FaceTrain()
                else:
                    SaySth("FaceData is not found, please contact adminstrator")
                    continue
            elif mod == "g":
                SaySth("get Face Feature, please keep you face in the frame and wait a minte")
                GetFeature()
         

if __name__ == '__main__':
    #main()
    pass
GetFeature()