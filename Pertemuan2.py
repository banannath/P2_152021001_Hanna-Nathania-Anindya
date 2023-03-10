import sys
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

# hide window tkinter
root = Tk()
root.withdraw()

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.image = None
        self.actionSave.triggered.connect(self.saveImage)
        self.pushButton.clicked.connect(self.fungsi)
        self.pushButton_2.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        # self.actionSave.triggered.connect(self.saveImage)

        self.horizontalSlider.valueChanged.connect(self.contrastSlider)
        self.horizontalSlider_2.valueChanged.connect(self.brightnessSlider)

        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.equalHistogram)

        self.actionTranslasi.triggered.connect(self.translasi)

        self.action_45_derajat.triggered.connect(self.rotasimin45)
        self.action45_derajat.triggered.connect(self.rotasi45)
        self.action_90_derajat.triggered.connect(self.rotasimin90)
        self.action90_derajat.triggered.connect(self.rotasi90)
        self.action180_derajat.triggered.connect(self.rotasi180)
        self.actionTranspose.triggered.connect(self.transpose)

        self.action4x.triggered.connect(self.resize4x)
        self.action3x.triggered.connect(self.resize3x)
        self.action2x.triggered.connect(self.resize2x)
        self.action3_4x.triggered.connect(self.resize3_4x)
        self.action1_2x.triggered.connect(self.resize1_2x)
        self.action1_4x.triggered.connect(self.resize1_4x)

    def fungsi(self):
        filename = askopenfilename()
        self.Image = cv2.imread(filename)
        self.displayImage(1)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j,2 ], 0 , 255)
        self.Image = gray
        self.displayImage(2)
        print(self.Image)

    def brightness(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        #looping berdasarkan lebar dan tinggi
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j) #self.Image.item(i, j) berarti mengakses nilai piksel (pixel value) pada koordinat baris (i) dan kolom (j) dari citra yang disimpan di dalam variabel self.Image.
                b = np.clip(a + brightness, 0, 255) #function untuk menjaga range si b tetap di antara 0-255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def contrastStretching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def negative(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = 255 - a

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def biner(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)

                if a == 180:
                    b = 0
                elif a < 180:
                    b = 1
                else:
                    b = 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def contrastSlider(self,value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = self.horizontalSlider.value()/100
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def brightnessSlider(self, value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = self.horizontalSlider_2.value()
        #looping berdasarkan lebar dan tinggi
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j) #self.Image.item(i, j) berarti mengakses nilai piksel (pixel value) pada koordinat baris (i) dan kolom (j) dari citra yang disimpan di dalam variabel self.Image.
                b = np.clip(a + brightness, 0, 255) #function untuk menjaga range si b tetap di antara 0-255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        print(self.Image)

    def grayHistogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2 ], 0 , 255)
        self.Image = gray
        print(self.Image)
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255]) #membuat histogram yang diperoleh dari pixel gambar dan memiliki nilai maksimum yaitu 255
        plt.show()

    def rgbHistogram(self) :
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
        plt.plot(histo, color=col)
        plt.xlim([0, 256])
        print(self.Image)
        plt.show()

    def equalHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() -
                                               cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    # Translasi
    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(2)
        print(self.Image)


    # Rotasi
    def rotasi90(self):
        self.rotasi(90)

    def rotasimin90(self):
        self.rotasi(-90)

    def rotasi45(self):
        self.rotasi(45)

    def rotasimin45(self):
        self.rotasi(-45)

    def rotasi180(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2),
                                                 degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h,w))
        self.Image = rot_image
        self.displayImage(2)
        print(self.Image)

    # Transpose
    def transpose(self):
        trans_image = cv2.transpose(self.Image)
        self.Image = trans_image
        self.displayImage(2)
        print(self.Image)

    # Resize
    def resize2x(self):
        self.resizedimensi(2)

    def resize3x(self):
        self.resizedimensi(3)

    def resize4x(self):
        self.resizedimensi(4)

    def resize1_2x(self):
        self.resizedimensi(0.5)

    def resize1_4x(self):
        self.resizedimensi(0.25)

    def resize3_4x(self):
        self.resizedimensi(0.75)

    def resizedimensi(self, scale):
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Original", self.Image)
        cv2.imshow("Resized", resize_img)
        cv2.waitKey()


    # Save Image
    def saveImage(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg *.jpeg);;PNG (*.png)")

        if not filename:
            return

        cv2.imwrite(filename, self.Image)
        QMessageBox.information(self, "Save Image", "Gambar berhasil disimpan!!")

    # Display Image
    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        if windows == 1:
            img = QImage(self.Image,
                     self.Image.shape[1],
                     self.Image.shape[0],
                     self.Image.strides[0], qformat)
            img = img.rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        elif windows == 2:
            img = QImage(self.Image,
                         self.Image.shape[1],
                         self.Image.shape[0],
                         self.Image.strides[0], qformat)
            img = img.rgbSwapped()
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

app=QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 2')
window.show()
sys.exit(app.exec_())
