from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect,HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib import messages

import cv2
import os
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import numpy as np


detector = MTCNN()

def home(request):
    # return HttpResponseRedirect('')
    # return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'index.html')

def process(request):

    if request.method == 'POST':
        myfile = request.FILES['pic']


        img = pyplot.imread(myfile)
        result_list = detector.detect_faces(img)

        # img = cv2.imread(myfile)
        # face_cascade = cv2.CascadeClassifier('media/haarcascade_frontalface_default.xml')
        # result_list = face_cascade.detectMultiScale(img, 1.3, 5)

       
        listdata = []
        ino = 1
        countpic = len(result_list)
        for i in range(len(result_list)):
            x1,y1, width, height = result_list[i]['box']
            x2,y2 = x1 + width, y1 + height
            lastface = img[y1:y2, x1:x2]
            lastimg = cv2.resize(lastface, (224, 224))
            cropface = cv2.cvtColor(lastimg, cv2.COLOR_BGR2RGB)

            textfile = str(myfile)
            # namesid = textfile[:-4]
            namesid = 'as'
            if ino == 1:
                savefile = '/media/' + namesid +'.png'
            else:
                savefile = '/media/' + namesid + '-'+str(ino)+'.png'
            
            ino += 1
            cv2.imwrite(savefile, cropface)
            
            # pyplot.imshow(lastface)
            # pyplot.show()
            
            listdata.append(cropface)

                
    else:
        return HttpResponseRedirect('/')
    return render(request, 'output.html',{'img':listdata , 'countpic':countpic})