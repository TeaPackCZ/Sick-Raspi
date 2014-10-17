# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import cv2.cv as cv
import math

class Digit_detect:
    def __init__(self, path, roi_size):
        self.path = path
        self.roisize = roi_size
        self.max_h=0
        self.min_h=0
        self.min_w=0
        self.keys=[ i for i in range(48,58)]
        self.keys.append(32)
        self.keys.append(97)
        self.samples =  np.empty((0,self.roisize*self.roisize))
        self.responses = []
        self.model = cv2.KNearest()
        self.element = np.ones([3,3],np.uint8)
        self.max_dist = 5000000

    def im_preprocess(self,image):
        size=image.shape
        self.max_h=size[0]*(2/3.0)
        self.min_h=size[0]/10
        self.min_w=size[1]/15
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        thresh = cv2.dilate(thresh, self.element, iterations = 3)
        thresh = cv2.erode(thresh, self.element, iterations = 3)
        return thresh

    def process_learning(self,image,binary):
        mybinary = np.copy(binary)
        contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>30:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if (w>self.min_w) and (h>self.min_h) and (w<h) and (h<self.max_h):
                    if True:#y<300:
                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                        roi = mybinary[y:y+h,x:x+w]
                        roismall = cv2.resize(roi,(self.roisize,self.roisize))
                        cv2.imshow('norm',image)
                        cv2.imshow('small',roismall)
                        key = cv2.waitKey(0)
                        if key == 27:  # (escape to skip rest of image)
                            break
                        elif key in self.keys:
                            if key == 32:
                                self.responses.append(20)
                            elif key == 97:
                                self.responses.append(10)
                            else:
                                self.responses.append(int(chr(key)))
                            sample = roismall.reshape((1,self.roisize*self.roisize))
                            self.samples = np.append(self.samples,sample,0)
    
    def learn_from_pic(self, path_to_files, clear_all, to_save, to_train):
        # do you want to start from zero?
            # True: clear all
            # False -> just continue
        if(clear_all):
            self.samples =  np.empty((0,self.roisize*self.roisize))
            self.responses = []
            self.model = 0
            self.model = cv2.KNearest()
        # make samples and responses
        cv2.namedWindow('norm',cv2.WINDOW_NORMAL)
        cv2.namedWindow('small',cv2.WINDOW_NORMAL)
        for img_path in path_to_files:
            imag = cv2.imread(img_path)
            binary = self.im_preprocess(imag)
            self.process_learning(imag,binary)
        cv2.destroyWindow('norm')
        cv2.destroyWindow('small')
        # ask for saving data
            # True -> save data
            # False -> nothing happens
        if(to_save):
            path_1 = self.path + 'samples_1.data'
            path_2 = self.path + 'responses_1.data'
            self.save_learned(path_1,path_2)
        # ask to train model on current datas
            # Y -> self.model.train(self.samples,self.responses)
            # N -> program ends
        if(to_train):
            responses = np.array(self.responses,np.float32)
            responses = responses.reshape((responses.size,1))
            samples = np.array(self.samples,np.float32)
            self.model.train(samples,responses)


    def save_learned(self,path_samples,path_responses):
        responses = np.array(self.responses,np.float32)
        responses = responses.reshape((responses.size,1))
        np.savetxt(path_samples,self.samples)
        np.savetxt(path_responses,responses)
        
            
    def learn_from_file(self,path_to_samples,path_to_responses):
        samples = np.loadtxt(path_to_samples,np.float32)
        responses = np.loadtxt(path_to_responses,np.float32)
        responses = responses.reshape((responses.size,1))
        self.model.train(samples,responses)

    def selection(self,points):
        points = sorted(points)#,reverse = True)
        detected = []
        ret = []
        for num in points:
            if num[0] not in detected:
                detected.append(num[0])
                ret.append(num)
        return ret

    def detect(self,binary):
        mybinary = np.copy(binary)
        contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        point=[]
        size = mybinary.shape
        for cnt in contours:
            if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if (w>self.min_w) and (h>self.min_h) and (w<h) and (h<self.max_h) and ( not((h<60) and (y<150)) ):
                    if True:#y<300:
                        roi = mybinary[y:y+h,x:x+w]
                        roismall = cv2.resize(roi,(self.roisize,self.roisize))
                        roismall = roismall.reshape((1,self.roisize*self.roisize))                                            
                        roismall = np.float32(roismall)
                        retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k=1)
                        if int((results[0][0]))<11 and dists<self.max_dist:
                            point.append([int(retval),int((dists)),x,y,w,h])
        best = self.selection(point)
        return best
        
    def detect_digits_from_file(self,path_to_image):
        digits = []
        image = cv2.imread(path_to_image)
        binary = self.im_preprocess(image)
        digits = self.detect(binary)
        return digits


    def detect_digits(self,cv_image):
        digits = []
        try:
            size = cv_image.shape
        except:
            return digits
        binary = self.im_preprocess(cv_image)
        digits = self.detect(binary)
        return digits

    def detect_digits_debug(self,src_paths,max_d = 10000000):
        cv2.namedWindow('norm',cv2.WINDOW_NORMAL)
        self.max_dist = max_d
        for path in src_paths:
            mydigits = []
            myimage = cv2.imread(path)
            size = myimage.shape
            mydigits = self.detect_digits(myimage)
            for digit in mydigits:
                cv2.putText(myimage,(str(digit[0]) + '-' + str(digit[1])),(int(digit[2]),int(digit[3])),1,1,(0,0,255))
                cv2.putText(myimage,('X:' + str(int(digit[2])) + ',Y:' + str(int(digit[3]))),(int(digit[2]+10),int(digit[3]+10)),1,1,(0,255,0))
            cv2.putText(myimage,path,(10,10),1,1,(0,0,255))
            cv2.imshow('norm',myimage)
            cv2.waitKey(0)
        cv2.destroyWindow('norm')

            
def make_paths(prefix,sufix,index_min,index_max):
    src = []
    for i in range(index_min,index_max):
        src.append(prefix + ('/%03d' %i) + sufix)
    return src
    

def view_data(path_smpls = 'samples_ver2.data', path_respnss = 'responses_ver2.data'):
    samples = np.loadtxt(path_smpls,np.float32)
    responses = np.loadtxt(path_respnss,np.float32)
    responses = responses.reshape((responses.size,1))
    samples = samples.reshape(responses.size,(samples.size/responses.size))
    index = 0
    new_samples = []
    new_responses = []
    cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
    for item in samples:
        if(int(responses[index]) < 11):
            print str(int(responses[index]))
            pic = item.reshape(math.sqrt(len(item)),math.sqrt(len(item)))
            cv2.imshow('preview',pic)
            key = cv2.waitKey(0);
            if key == 32:
                new_samples.append(item)
                new_responses.append(responses[index])
        index+=1
    cv2.destroyWindow('preview')
    if(to_save == 1):
        np.savetxt('new_samples',new_samples)
        np.savetxt('new_responses',new_responses)
        print('Data saved')
    print ('End ...')

def append_data(samples_1, responses1, samples2, responses2,skip_1 = False, skip_2 = False):
    samples1 = np.loadtxt(samples1,np.float32)
    responses1 = np.loadtxt(responses1,np.float32)
    responses1 = responses1.reshape((responses1.size,1))
    samples1 = samples1.reshape(responses1.size,(samples1.size/responses1.size))

    samples2 = np.loadtxt(samples2,np.float32)
    responses2 = np.loadtxt(responses2,np.float32)
    responses2 = responses2.reshape((responses2.size,1))
    samples2 = samples2.reshape(responses2.size,(samples2.size/responses2.size))

    if(samples1.shape is not samples2.shape):
        print ('Data are not compatible, please select samples with same size')
        return -1
    
    #print samples.shape
    new_samples = []
    new_responses = []
    cv2.namedWindow('Add shape?',cv2.WINDOW_NORMAL)
    
    # First samples & shapes
    if(skip_1):
        new_samples = samples1
        new_responses = responses1
    else:
        index = 0
        for item in samples1:
            if(int(responses1[index]) < 11):
                print str(int(responses1[index]))
                pic = item.reshape(math.sqrt(len(item)),math.sqrt(len(item)))
                cv2.imshow('Add shape?',pic)
                key = cv2.waitKey(0);
                if key == 32:
                    new_samples.append(item)
                    new_responses.append(responses1[index])
            index+=1

    # Second samples & shapes
    index = 0
    for item in samples2:
        if(int(responses2[index]) < 11):
            if(skip_2 is False):
                print str(int(responses2[index]))
                pic = item.reshape(math.sqrt(len(item)),math.sqrt(len(item)))
                cv2.imshow('Add shape?',pic)
                key = cv2.waitKey(0)
            else:
                key=32
            if key == 32:
                new_samples.append(item)
                new_responses.append(responses2[index])
        index+=1
            
    cv2.destroyWindow('Add shape?')
    if(to_save == 1):
        np.savetxt('new_samples',new_samples)
        np.savetxt('new_responses',new_responses)
        print('Data saved')
    print ('End ...')
