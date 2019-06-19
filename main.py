#!/usr/bin/env python
# coding: utf-8

# In[4]:


import Inpainter as pt
import sys
import cv2 as cv
import os
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# In[5]:


def cv_imread(filePath):
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

class gui(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        
    def initUI(self):      
        self.btn = QPushButton('image', self)
        self.btn.setToolTip('请选择你的文件')
        self.btn.clicked.connect(self.file_op_1)
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(30,10)
        
        self.btn2 = QPushButton('mask', self)
        self.btn2.setToolTip('请选择你的文件')
        self.btn2.clicked.connect(self.file_op_2)
        self.btn2.resize(self.btn2.sizeHint())
        self.btn2.move(30,50)
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        #grid.addWidget(self.btn, 1, 0)
        #grid.addWidget(self.btn2, 1, 1)
        word = QLabel('可选参数：',self)
        word.move(10,130)
        
        self.btn3 = QPushButton('patch_size', self)
        self.btn3.clicked.connect(self.op_3)
        self.btn3.resize(self.btn3.sizeHint())
        self.btn3.move(30,160)
        
        self.btn4 = QPushButton('save_position', self)
        self.btn4.setToolTip('结果文件保存位置')
        self.btn4.clicked.connect(self.op_4)
        self.btn4.resize(self.btn4.sizeHint())
        self.btn4.move(30,90)
        
        self.btn5 = QPushButton('GO!!!', self)
        self.btn5.clicked.connect(self.solve)
        self.btn5.resize(self.btn5.sizeHint())
        self.btn5.move(100,220)
        
        self.word2 = QLabel('',self)
        self.word2.move(100,250)
        
        self.btn6 = QPushButton('show_process?', self)
        self.btn6.clicked.connect(self.op_6)
        self.btn6.resize(self.btn6.sizeHint())
        self.btn6.move(140,160)
        
        self.setLayout(grid)
        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('Inpainting')
        self.setWindowIcon(QIcon('mask.jpg')) 
        
        self.center()
        self.show()
        
        self.my_image = None
        self.my_mask = None
        self.patch_size = 9
        self.my_path = os.getcwd()
        self.my_show = False
        
    def file_op_1(self):
        cwd=os.getcwd()
        name = QFileDialog.getOpenFileName(self, 'Open file', cwd)
        self.my_image = cv_imread(name[0])
        self.btn.setStyleSheet('''QPushButton{background-color : green;}''')
            
    def file_op_2(self):
        cwd=os.getcwd()
        name = QFileDialog.getOpenFileName(self, 'Open file', cwd)
        self.my_mask = cv_imread(name[0])
        self.btn2.setStyleSheet('''QPushButton{background-color : green;}''')
        
    def op_3(self):
        text, ok = QInputDialog.getText(self, 'Inputing patch_size', 
            'input the patch_size(9 default):')
        if ok:
            self.patch_size = int(text)
            self.btn3.setText(text)
            
    def op_6(self):
        self.my_show = True
        self.btn6.setText('show_process!')
        
    def op_4(self):
        cwd=os.getcwd()
        name = QFileDialog.getSaveFileName(self, 'Save file', cwd)
        self.my_path = name[0]+'.jpg'
        self.btn4.setStyleSheet('''QPushButton{background-color : green;}''')
 
    def closeEvent(self, event):#重载closeEvent
        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()  
    
    def center(self):
        
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def solve(self):
        self.word2.setText('Solveing~~~~~~~~')
        Alice  = pt.Inpainter(self.my_image,self.my_mask,
                              patch_size = self.patch_size,
                              show = self.my_show
                             )
        img = Alice.solve()
        self.word2.setText('OK!')
        cv.imencode('.jpg', img_cv)[1].tofile(self.my_path)
        


# In[6]:


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = gui()
    sys.exit(app.exec_())


# In[ ]:




