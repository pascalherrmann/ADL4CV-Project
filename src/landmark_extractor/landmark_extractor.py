"""
Created on Fri May 15 15:21:29 2020

@author: Tobias Zengerle
"""
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import csv

from landmark_extractor.face_alignment import FaceAlignment, LandmarksType

class FaceLandmarkExtractor:
    def __init__(self):
        self.face_aligner = FaceAlignment(LandmarksType._2D, flip_input=False, device = 'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def extract_landmarks(self, source_path_or_image):
        try:
            return self.face_aligner.get_landmarks_from_image(source_path_or_image)[0]
        except:
            print('Error: couldnt extract landmarks')
            return None
    
    def generate_landmark_image(self, source_path_or_image, output_image_path='', keypoint_csv_dir='', resolution=128):
        try:
            preds = self.extract_landmarks(source_path_or_image)
            input_shape = np.zeros((resolution,resolution))
            dpi = 100
            fig = plt.figure(figsize=(input_shape.shape[1]/dpi, input_shape.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            
             # show the background image
            # create a  background image
            #X = np.linspace(0, np.pi, 100)
            img = np.zeros([resolution, resolution, 3], dtype=np.uint8)#np.sin(X[:,None] + X[None,:])
            #x0,x1 = ax.get_ylim()
            #y0,y1 = ax.get_xlim()
            ax.imshow(img, vmin=0, vmax=255)#, extent=[x0, x1, y0, y1], aspect='auto')
            
            #ax.imshow(np.zeros(input_shape.shape, dtype=np.uint8))#np.zeros(input_shape.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=1)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=1)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=1)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=1)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=1)
            #left and right eye
            ax.plot(np.concatenate((preds[36:42,0], np.array([preds[36,0]]))),np.concatenate((preds[36:42,1], np.array([preds[36,1]]))),marker='',markersize=5,linestyle='-',color='red',lw=1)
            ax.plot(np.concatenate((preds[42:48,0], np.array([preds[42,0]]))),np.concatenate((preds[42:48,1], np.array([preds[42,1]]))),marker='',markersize=5,linestyle='-',color='red',lw=1)
            #outer and inner lip
            #ax.plot(np.concatenate((preds[48:60,0], np.array([preds[48,0]]))), np.concatenate((preds[48:60,1], np.array([preds[48,1]]))),marker='',markersize=5,linestyle='-',color='purple',lw=1)
            ax.plot(np.concatenate((preds[60:68,0], np.array([preds[60,0]]))),np.concatenate((preds[60:68,1], np.array([preds[60,1]]))),marker='',markersize=5,linestyle='-',color='pink',lw=1) 
            ax.axis('off')
            
            fig.canvas.draw()
            
            if output_image_path != '':
                fig.savefig(output_image_path)
                
            if keypoint_csv_dir != '':
                with open(keypoint_csv_dir, 'w', newline='') as csvfile:
                    fieldnames = ['keypoints']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                    writer.writeheader()
                    writer.writerow({'keypoints': preds})
    
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
        except:
            print('Error: Image corrupted or no landmarks visible')
            return None
        
        data = torch.from_numpy(np.array(data)).type(dtype = torch.float)
        return data, preds
    
    
    def create_landmark_dataset(self, source_root='', output_root='', keypoint_csv_dir='', resolution=128):
        for subdir, dirs, files in os.walk(source_root):
            for file in files:
                out_dir = os.path.join(output_root, os.path.relpath(subdir, source_root))
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                    
                csv_dir = os.path.join(keypoint_csv_dir, os.path.relpath(subdir, source_root))
                if not os.path.isdir(csv_dir):
                    os.makedirs(csv_dir)                    
                self.generate_landmark_image(os.path.join(subdir, file), os.path.join(out_dir, file), os.path.join(csv_dir, file), resolution)
    
    def create_keypoints_only(self, source_root='', keypoint_csv_dir='', resolution=128):
        for subdir, dirs, files in os.walk(source_root):
            for file in files:
                csv_dir = os.path.join(keypoint_csv_dir, os.path.relpath(subdir, source_root))
                if not os.path.isdir(csv_dir):
                    os.makedirs(csv_dir)   
                self.generate_landmark_image(os.path.join(subdir, file), '', os.path.join(csv_dir, file), resolution)