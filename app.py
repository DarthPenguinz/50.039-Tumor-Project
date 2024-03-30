import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from functions import *
from UNET import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import __main__

model_path = "./saved_models/multi_channel_main/model_5.pth"
setattr(__main__, "UNet", UNet)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path, map_location=torch.device(device))

import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_slices(segmentation_3d, input_mri_3d):
    # Check if the input arrays have the same shape
    assert segmentation_3d.shape == input_mri_3d.shape, "The input arrays must have the same shape."
    
    # Convert inputs to PyTorch tensors if they're not already
    segmentation_3d = torch.tensor(segmentation_3d)
    input_mri_3d = torch.tensor(input_mri_3d)
    
    # Initialize a list to hold processed slices
    processed_slices = []
    
    # Iterate over each slice
    for i in range(segmentation_3d.shape[2]):
        # Extract the current slice
        segmentation_slice = segmentation_3d[:, :, i]
        mri_slice = input_mri_3d[:, :, i]
        
        # Threshold the segmentation slice to get a binary mask
        thresholded_slice = (segmentation_slice > 0).to(torch.uint8) * 255
        
        # Simplified contour finding (just to demonstrate, this part needs more work for actual contour detection)
        # Find indices where there are changes (edges in the binary image), which we'll use as a simple contour approximation
        
        # For bounding box calculation: Find min and max of the thresholded indices
        if thresholded_slice.sum() > 0:  # Check if there is any segmentation
            nonzero_indices = torch.nonzero(thresholded_slice)
            min_y, max_y = torch.min(nonzero_indices[:, 0]), torch.max(nonzero_indices[:, 0])
            min_x, max_x = torch.min(nonzero_indices[:, 1]), torch.max(nonzero_indices[:, 1])
            
            # Convert MRI slice to 3-channel image by repeating it across a new dimension
            if mri_slice.dtype != torch.uint8:
                mri_slice = (mri_slice * 255).to(torch.uint8)
            mri_slice_bgr = mri_slice.repeat(3, 1, 1).permute(1, 2, 0)  # Shape: Height x Width x Channels
            
            # Draw bounding box
            mri_slice_bgr[min_y:max_y, min_x] = torch.tensor([0, 255, 0])  # Left line
            mri_slice_bgr[min_y:max_y, max_x] = torch.tensor([0, 255, 0])  # Right line
            mri_slice_bgr[min_y, min_x:max_x] = torch.tensor([0, 255, 0])  # Top line
            mri_slice_bgr[max_y, min_x:max_x] = torch.tensor([0, 255, 0])  # Bottom line
            
            processed_slices.append(mri_slice_bgr.numpy())
    
    if processed_slices:
        # Stack and transpose to move channels to the last dimension
        array = np.stack(processed_slices).transpose(1, 2, 0, 3)
    else:
        # No processed slices, return an empty array with the correct shape
        array = np.empty((0, 0, 0, 3), dtype=np.uint8)
    
    return array



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, cmap="gray"):
        self.cmap = cmap
        fig = Figure(dpi=dpi, frameon=False)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        super(MplCanvas, self).__init__(fig)
    
    def display_image(self, img, width=5, height=4):
        """
        Displays an image on the canvas without showing the axes and with minimal border,
        and sets a fixed size for the canvas based on the image size.

        Parameters:
        - img: A 2D numpy array representing the image to display.
        - width: Width of the image in inches.
        - height: Height of the image in inches.
        """
        # Clear the axes for the new image
        self.axes.clear()
        self.axes.axis('off')

        # Display the image
        self.axes.imshow(img, cmap=self.cmap, aspect='auto')
        self.draw()
        
class MPLWidget(QWidget):
    def __init__(self, canvas, size):
        super().__init__()
        # Set up a layout and add the canvas to it
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        
        self.setLayout(layout)
        self.setFixedSize(size, size)  # Adjust these values based on your requirements


class FileUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Placeholder for storing file paths
        self.file_paths = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None}
        self.file_data = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None}
        self.file_layer = {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50 , 7: 50, 8: 50, 9: 50}

    def initUI(self):
        self.setFixedSize(1000, 900)
        self.layout = QVBoxLayout()
        self.inputslayout = QHBoxLayout()
        self.outputslayout = QHBoxLayout()
        self.boxedoutputslayout = QHBoxLayout()
        
        self.button1layout = QVBoxLayout()
        self.button2layout = QVBoxLayout()
        self.button3layout = QVBoxLayout()
        self.button4layout = QVBoxLayout()
        
        mrisize = 200
        slidersize = 200
        
        
        # Create upload buttons and connect them
        
        # Button1
        self.uploadBtn1 = QPushButton('Flair')
        self.uploadBtn1.clicked.connect(lambda: self.openFileNameDialog(0))
        self.display1 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display1canvas = MPLWidget(self.display1,mrisize)
                    
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)
        self.slider1.setFixedWidth(slidersize)
        self.slider1.valueChanged.connect(lambda: self.updateLabel(self.slider1,0))
        
        self.button1layout.addWidget(self.uploadBtn1)
        self.button1layout.addWidget(self.slider1)
        self.button1layout.addWidget(self.display1canvas)
        
        # Button2
        self.uploadBtn2 = QPushButton('T1')
        self.uploadBtn2.clicked.connect(lambda: self.openFileNameDialog(1))
        self.display2 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display2canvas = MPLWidget(self.display2,mrisize)
        
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)
        self.slider2.setValue(50)
        self.slider2.setFixedWidth(slidersize)
        self.slider2.valueChanged.connect(lambda: self.updateLabel(self.slider2,1))
        
        self.button2layout.addWidget(self.uploadBtn2)
        self.button2layout.addWidget(self.slider2)
        self.button2layout.addWidget(self.display2canvas,mrisize)
        
        # Button3
        self.uploadBtn3 = QPushButton('T1CE')
        self.uploadBtn3.clicked.connect(lambda: self.openFileNameDialog(2))
        self.display3 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display3canvas = MPLWidget(self.display3,mrisize)
        
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setMinimum(0)
        self.slider3.setMaximum(100)
        self.slider3.setValue(50)
        self.slider3.setFixedWidth(slidersize)
        self.slider3.valueChanged.connect(lambda: self.updateLabel(self.slider3,2))
        
        self.button3layout.addWidget(self.uploadBtn3)
        self.button3layout.addWidget(self.slider3)
        self.button3layout.addWidget(self.display3canvas,mrisize)

        # Button 4
        self.uploadBtn4 = QPushButton('T2')
        self.uploadBtn4.clicked.connect(lambda: self.openFileNameDialog(3))
        self.display4 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display4canvas = MPLWidget(self.display4,mrisize)
        
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setMinimum(0)
        self.slider4.setMaximum(100)
        self.slider4.setValue(50)
        self.slider4.setFixedWidth(slidersize)
        self.slider4.valueChanged.connect(lambda: self.updateLabel(self.slider4,3))
        
        self.button4layout.addWidget(self.uploadBtn4)
        self.button4layout.addWidget(self.slider4)
        self.button4layout.addWidget(self.display4canvas)
        
        self.inputslayout.addLayout(self.button1layout)
        self.inputslayout.addLayout(self.button2layout)
        self.inputslayout.addLayout(self.button3layout)
        self.inputslayout.addLayout(self.button4layout)
        self.layout.addLayout(self.inputslayout)
        
        
        self.output1layout = QVBoxLayout()
        self.output2layout = QVBoxLayout()
        
        # Create the analyze button
        self.analyzeBtn = QPushButton('Analyze')
        self.displayoutput = MplCanvas(self, width=5, height=5, dpi=240)
        self.display5canvas = MPLWidget(self.displayoutput,mrisize)

        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setMinimum(0)
        self.slider5.setMaximum(100)
        self.slider5.setValue(50)
        self.slider5.setFixedWidth(slidersize)
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,4))
        
        self.analyzeBtn.clicked.connect(self.analyze)
        
        
        self.layout.addWidget(self.analyzeBtn)
        
        self.output1layout.addWidget(self.slider5)
        self.output1layout.addWidget(self.display5canvas)
        
        self.uploadBtn6 = QPushButton('Upload Mask')
        self.uploadBtn6.clicked.connect(lambda: self.openFileNameDialog(5))
        self.display6 = MplCanvas(self, width=5, height=5, dpi=240, cmap="gray")
        self.display6canvas = MPLWidget(self.display6,mrisize)
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,5))
        
        self.output2layout.addWidget(self.uploadBtn6)
        self.output2layout.addWidget(self.display6canvas)
        
        self.outputslayout.addLayout(self.output1layout)
        self.outputslayout.addLayout(self.output2layout)
        
        self.outputlap1 = QVBoxLayout()
        self.outputlap2 = QVBoxLayout()
        self.outputlap3 = QVBoxLayout()
        self.outputlap4 = QVBoxLayout()
        
        self.display7 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display7canvas = MPLWidget(self.display7,mrisize)
                    
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,6))
        self.outputlap1.addWidget(self.display7canvas)
        
        # Button2
        self.display8 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display8canvas = MPLWidget(self.display8,mrisize)
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,7))
        
        self.outputlap2.addWidget(self.display8canvas)
        
        # Button3
        self.display9 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display9canvas = MPLWidget(self.display9,mrisize)
        
        self.slider9 = QSlider(Qt.Horizontal)
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,8))
        
        self.outputlap3.addWidget(self.display9canvas)

        # Button 4

        self.display10 = MplCanvas(self, width=5, height=5, dpi=240)
        self.display10canvas = MPLWidget(self.display10,mrisize)
        self.slider5.valueChanged.connect(lambda: self.updateLabel(self.slider5,9))
        
        self.outputlap4.addWidget(self.display10canvas)
        
        self.inputslayout.addLayout(self.button1layout)
        self.inputslayout.addLayout(self.button2layout)
        self.inputslayout.addLayout(self.button3layout)
        self.inputslayout.addLayout(self.button4layout)
        
        self.boxedoutputslayout.addLayout(self.outputlap1)
        self.boxedoutputslayout.addLayout(self.outputlap2)
        self.boxedoutputslayout.addLayout(self.outputlap3)
        self.boxedoutputslayout.addLayout(self.outputlap4)
        
        
        self.layout.addLayout(self.inputslayout)
        self.layout.addLayout(self.outputslayout)
        self.layout.addLayout(self.boxedoutputslayout)
        

        # Placeholder for image output
        self.imageLabel = QLabel(self)
        self.layout.addWidget(self.imageLabel)

        self.setLayout(self.layout)
        self.setWindowTitle('File Uploader and Analyzer')
        
        self.displays = {0: self.display1, 1: self.display2, 2: self.display3, 3: self.display4, 4: self.displayoutput, 5: self.display6, 6: self.display7, 7: self.display8, 8: self.display9, 9: self.display10}

    def openFileNameDialog(self, button_id):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            if fileName.endswith('.nii'):
                print(f"File path chosen: {fileName}")
                self.file_paths[button_id] = fileName
                self.process_file(fileName, button_id)

    def process_file(self, file_path, button_id):
        # Implement your file processing here
        print(f"Processing file {button_id}: {file_path}")
        mri = nib.load(file_path).get_fdata()
        mri = mri.copy()
        self.file_data[button_id] = mri
        self.display(button_id)
        
    def display(self,button_id):
        mri = self.file_data[button_id]
        if mri is not None:
            layer = (mri.shape[2] * self.file_layer[button_id]) // 100
            mri = mri[:,:,min(layer, mri.shape[2]-1)]
            self.displays[button_id].display_image(mri)
        
    def updateLabel(self, slider, button_id):
        value = slider.value()
        self.file_layer[button_id] = value
        self.display(button_id)
        
        
    def analyze(self):
        if self.file_paths[0] is None or self.file_paths[1] is None or self.file_paths[2] is None or self.file_paths[3] is None:
            print("Please upload all files")
            return
        mri_scans = [self.file_data[0], self.file_data[1], self.file_data[2], self.file_data[3]]
        mri_scans = np.array(mri_scans)
        mri_scans = torch.from_numpy(mri_scans).float()
        mri_scans = mri_scans.unsqueeze(0)
        mri_scans.to(device)
        model.to(device)
        
        output = []
        
        for i in range(mri_scans.size(-1)):
            slice_squeezed = mri_scans[..., i]
            slice_squeezed.to(device)  # This will have shape [1, 1, 240, 240]
            layer_pred = model(slice_squeezed)
            output.append(layer_pred[0][0].cpu().detach().numpy())
        print("Done analysing")
        self.file_data[4] = np.transpose(np.array(output), (1, 2, 0))
        self.file_data[6] = process_slices(self.file_data[4], self.file_data[0])
        self.file_data[7] = process_slices(self.file_data[4], self.file_data[1])
        self.file_data[8] = process_slices(self.file_data[4], self.file_data[2])
        self.file_data[9] = process_slices(self.file_data[4], self.file_data[3])
        self.display(4)
        self.display(6)
        self.display(7)
        self.display(8)
        self.display(9)
        
        
        pixmap = QPixmap('output.png')
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.resize(pixmap.width(), pixmap.height())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileUploader()
    ex.show()
    sys.exit(app.exec_())
