import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random
import math

class Inpainter():
    def __init__(self, image, mask, patch_size = 9,show=True):
        #image (rgb) mask (1 need fill; 0 not)
        self.image = image.astype('uint8')
        self.gray_image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY).astype('float')/255
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        self.mask = (cv.GaussianBlur(mask.astype('uint8'),(3,3),1.5) > 0).astype('uint8')
        self.fill_image = self.image.copy()
        self.fill_image = self.fill_image*((1-self.mask)[:,:,np.newaxis].repeat(3,axis=2))
        self.fill_range = self.mask.copy()
        self.patch_size = patch_size
        self.priority, self.C, self.D, self.tmp = [np.zeros_like(self.gray_image) for i in range(4)]
        self.C += (mask == 0)
        self.shape = self.mask.shape
        self.h,self.w = mask.shape
        self.show = show
        self.target = None
        self.isophote = None
        self.normal = None
        self.gradient = None
        self.diff_img = None
        self.init_front = None
        self.sample = None
        self.todo_num = 0
        self.now_num = 0
    def solve(self):
        #do the things
        self.get_sample()
        self.todo_num = self.fill_range.sum()
        while self.fill_range.sum():
            #print(self.fill_range.sum())
            self.now_num = self.fill_range.sum()
            self.get_front()
            self.update_priority()
            self.target = self.get_target()
            lab_image = cv.cvtColor(self.fill_image, cv.COLOR_RGB2Lab)
            
            #self.diff_img = cv.GaussianBlur(self.fill_image,(3,3),5)
            #self.diff_img = cv.cvtColor(cv.GaussianBlur(self.fill_image,(3,3),1.5), cv.COLOR_RGB2Lab)
            self.diff_img = lab_image
            
            self.front_fill()
            if self.show == True:
                cv.imshow('img',self.fill_image)
                cv.waitKey(30)
        
        return self.fill_image
        
    def get_sample(self):
        self.init_front = (cv.Laplacian(self.fill_range, -1) > 0).astype('uint8')
        self.sample = np.zeros_like(self.gray_image).astype('uint8')
        for p in np.argwhere(self.init_front == 1):
            self.patch_data(self.sample, self.get_range(p,int(self.patch_size)))[:] = 1
            
        half = self.patch_size//2
        tmp = np.zeros_like(self.sample)
        self.patch_data(tmp,[[half,self.h-half],[half,self.w-half]])[:] = 1
        self.sample*=tmp
    def silly_fill(self):
        half = self.patch_size//2
        self.do_fill((random.randint(half,self.h-half-1),random.randint(half,self.w-half-1)))
    def rand_fill(self):
        half = self.patch_size//2
        self.tmp[::] = 10000000000000000000000000000000000000
        for t in range(1000):
            i,j = random.randint(half,self.h-half-1),random.randint(half,self.w-half-1)
            if self.patch_data(self.fill_range, self.get_range((i,j))).sum() == 0:
                self.tmp[i,j]=self.diff(self.target,(i,j))
        pos = np.unravel_index(self.tmp.argmin(),self.tmp.shape)
        self.do_fill(pos)
    def fill(self):
        half = self.patch_size//2
        self.tmp[::] = 10000000000000000000000000000000000000
        for i in range(half,self.h-half):
            for j in range(half,self.w-half):
                if self.patch_data(self.mask, self.get_range((i,j))).sum() == 0:
                    self.tmp[i,j]=self.diff(self.target,(i,j))
        pos = np.unravel_index(self.tmp.argmin(),self.tmp.shape)
        self.do_fill(pos)
    def front_fill(self):
        half = self.patch_size//2
        self.tmp[::] = 10000000000000000000000000000000000000
        for p in np.argwhere(self.sample == 1):
            if self.patch_data(self.mask, self.get_range(p)).sum() == 0:
                self.tmp[p[0],p[1]]=self.diff(self.target, p)
        pos = np.unravel_index(self.tmp.argmin(),self.tmp.shape)
        #self.diff_3(self.target, pos)
        self.do_fill(pos)
    def local_fill(self):
        half = self.patch_size//2
        self.tmp[::] = 10000000000000000000000000000000000000
        rg = self.get_range(self.target, 2*self.patch_size)
        for i in range(max(half,rg[0][0]),min(self.h-half,rg[0][1])):
            for j in range(max(half,rg[1][0]),min(self.w-half,rg[1][1])):
                if self.patch_data(self.fill_range, self.get_range((i,j))).sum() == 0:
                    self.tmp[i,j]=self.diff(self.target,(i,j))
        pos = np.unravel_index(self.tmp.argmin(),self.tmp.shape)
        self.do_fill(pos)
    def do_fill(self,pos):
        p1,p2 = self.target,pos
        #print(p1,p2)
        goal = np.where(self.patch_data(self.fill_range, self.get_range(p1)) > 0)
        rg1 = self.get_range(p1)
        t1 = self.patch_data(self.fill_image, rg1)
        t2 = self.patch_data(self.fill_image, self.fit_range(p1,rg1,p2))
        t1[goal[0],goal[1]] = t2[goal[0],goal[1]]
        c = self.patch_data(self.C, self.get_range(p1))
        c[goal[0],goal[1]] = self.C[p1[0],p1[1]]
        self.patch_data(self.fill_range, self.get_range(p1))[:] = 0
    def diff_2(self,p1,p2):
        rg1 = self.get_range(p1)
        rg2 = self.fit_range(p1,rg1,p2)
        mask = (1-self.patch_data(self.fill_range,rg1))[:,:,np.newaxis].repeat(3,axis = 2)
        t1 = self.patch_data(self.diff_img, rg1)*mask
        t2 = self.patch_data(self.diff_img, rg2)*mask
        dist = math.sqrt(((np.array(p1)-np.array(p2))**2).sum())
        alpha = self.now_num/self.todo_num
        return ((t1-t2)**2).sum() 
        #+ alpha*alpha*4*dist*dist
     ##############################################################################################################
    
    def diff_4(self,p1,p2):
        rg1 = self.get_range(p1)
        rg2 = self.fit_range(p1,rg1,p2)
        mask = (1-self.patch_data(self.fill_range,rg1))
        mask3 = mask[:,:,np.newaxis].repeat(3,axis = 2)
        t1 = self.patch_data(self.diff_img, rg1)*mask3
        t2 = self.patch_data(self.diff_img, rg2)*mask3
        c1 = np.sum(np.abs(self.patch_data(self.gradient, rg1)*mask))
        c2 = np.sum(np.abs(self.patch_data(self.gradient, rg2)*mask))
        tmp = ((t1-t2)**2).sum() 
        dist = math.sqrt(((np.array(p1)-np.array(p2))**2).sum())
        alpha = self.now_num/self.todo_num
        #print(alpha*alpha*4*dist)
        #print(tmp)
        return tmp + abs(c1-c2)*1000  + alpha*alpha*20*dist
        #if abs(c1-c2)<0.1 else 1000000000000000000
        
    def diff_3(self,p1,p2):
        rg1 = self.get_range(p1)
        rg2 = self.fit_range(p1,rg1,p2)
        mask = (1-self.patch_data(self.fill_range,rg1))
        mask3 = mask[:,:,np.newaxis].repeat(3,axis = 2)
        t1 = self.patch_data(self.diff_img, rg1)*mask3
        t2 = self.patch_data(self.diff_img, rg2)*mask3
        
        tmp = ((t1-t2)**2).sum()
        
        t3 = (self.patch_data(self.diff_img, rg2)*(1-mask3)).transpose([2,1,0])
        t4 = t1.transpose([2,1,0])
        dt = mask.sum()
        dt_2 = (1-mask).sum() 
        tmp_2 = sum([(t3[i,:,:].sum()/dt_2 - t4[i,:,:].sum()/dt)**2 for i in range(3)])
        
        print(tmp_2)
        return tmp
    
    def diff(self,p1,p2):
        rg1 = self.get_range(p1)
        rg2 = self.fit_range(p1,rg1,p2)
        mask = (1-self.patch_data(self.fill_range,rg1))
        mask3 = mask[:,:,np.newaxis].repeat(3,axis = 2)
        t1 = self.patch_data(self.diff_img, rg1)*mask3
        t2 = self.patch_data(self.diff_img, rg2)*mask3
        
        tmp = ((t1-t2)**2).sum()
        
        t3 = (self.patch_data(self.diff_img, rg2)*(1-mask3)).transpose([2,1,0])
        t4 = t1.transpose([2,1,0])
        dt = mask.sum()+1
        dt_2 = (1-mask).sum()+1
        tmp_2 = sum([(t3[i,:,:].sum()/dt_2 - t4[i,:,:].sum()/dt)**2 for i in range(3)])
        
        dist = math.sqrt(((np.array(p1)-np.array(p2))**2).sum())
        alpha = self.now_num/self.todo_num
        
        return tmp + tmp_2 + alpha*alpha*dist*dist
        
    def get_target(self):
        return np.unravel_index(self.priority.argmax(), self.priority.shape)
    def update_C(self):
        front_p = np.argwhere(self.front == 1)
        n_C = self.C.copy()
        for p in front_p:
            n_C[p[0],p[1]] = self.patch_data(self.C, self.get_range(p)).sum()/self.range_area(p)
        self.C = n_C +  + 0.0000000000001
    def get_normal(self):
        x,y = cv.Scharr(self.fill_range,cv.CV_64F,1,0),cv.Scharr(self.fill_range,cv.CV_64F,0,1)
        normal = np.dstack([x,y])
        norm = np.sqrt(x**2 + y**2).reshape(self.shape[0],self.shape[1],1).repeat(2,axis=2)
        norm[norm==0] = 1
        self.normal = normal/norm
    def get_isophote(self):
        gray_image = cv.cvtColor(self.fill_image, cv.COLOR_RGB2GRAY).astype('float')/255
        gray_image[self.fill_range == 1] = None
        gradient = np.nan_to_num(np.array(np.gradient(gray_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        self.gradient = gradient_val
        max_gradient = np.zeros([self.h, self.w, 2])
        front_p = np.argwhere(self.front == 1)
        for p in front_p:
            max_gradient[p[0], p[1], 0] = -gradient[0,p[0],p[1]]
            max_gradient[p[0], p[1], 1] = gradient[1,p[0],p[1]]
        self.isophote = max_gradient
        
    def get_isophote_2(self):
        gray_image = cv.cvtColor(self.fill_image, cv.COLOR_RGB2GRAY).astype('float')/255
        gray_image[self.fill_range == 1] = None
        gradient = np.nan_to_num(np.array(np.gradient(gray_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        self.gradient = gradient_val
        max_gradient = np.zeros([self.h, self.w, 2])
        front_p = np.argwhere(self.front == 1)
        for p in front_p:
            patch = self.get_range(p)
            y = self.patch_data(gradient[0], patch)
            x = self.patch_data(gradient[1], patch)
            val = self.patch_data(gradient_val, patch)
            pos = np.unravel_index(val.argmax(),val.shape)
            max_gradient[p[0], p[1], 0] = -y[pos]
            max_gradient[p[0], p[1], 1] = x[pos]
        self.isophote = max_gradient
        
    def update_D(self):
        self.get_normal()
        self.get_isophote()
        self.D = abs(self.normal[:,:,0]*self.isophote[:,:,0] + self.normal[:,:,1]*self.isophote[:,:,1])
        
        self.D = self.D + 0.00000001
        #self.D = abs(self.normal[:, :, 0]*self.isophote[:, :, 0]**2 +self.normal[:, :, 1]*self.isophote[:, :, 1]**2)+0.001
    def calc_priority(self):
        self.priority = self.front * self.C * self.D
        ##############################################################################################################
    def update_priority(self):
        self.update_C()
        self.update_D()
        self.calc_priority()
    def get_front(self):
        self.front = (cv.Laplacian(self.fill_range, -1) > 0).astype('uint8')
    def get_range(self, p, l=-1):
        if l==-1:
            l=self.patch_size
        half = l//2
        return np.array([[max(0,p[i]-half),min(p[i]+half+1,self.shape[i])] for i in range(2)])
    def fit_range(self,p1,rg,p2):
        return np.array([[p2[i]-(p1[i]-rg[i][0]),p2[i]-(p1[i]-rg[i][1])] for i in range(2)])
    def range_area(self, p, l=-1):
        if l==-1:
            l=self.patch_size
        area = self.get_range(p,l)
        return (area[0][1] - area[0][0])*(area[1][1] - area[1][0])
    @staticmethod
    def patch_data(img, patch_range):
        return img[patch_range[0][0]:patch_range[0][1], patch_range[1][0]:patch_range[1][1]]

