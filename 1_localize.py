
import cv2
import os
import pandas as pd
import numpy as np
import random
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import json

import utils.calc as calc

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
colors = {
    'noncon': RED,
    'h_rect': GREEN,
    'square': BLUE,
    'v_rect': YELLOW,
}
shapes = ['noncon', 'h_rect', 'square', 'v_rect']

def load_data(points_path, voting_path):

    # load points
    data = {}
    for file in os.listdir(points_path):
        point_df = pd.read_csv(os.path.join(points_path, file))
        for index, row in point_df.iterrows():
            if not row['filename'] in data:
                data[row['filename']] = {
                    'points': [[0, 0]] * 19,
                    'voting': [0, 0, 0, 0]
                }
            d = eval(row['region_shape_attributes'])
            x, y = int(d['cx']), int(d['cy'])
            data[row['filename']]['points'][row['region_id']] = [x, y]

    # load voting
    voting_df = pd.read_csv(voting_path)
    for index, row in voting_df.iterrows():
        filename = str(row['No.']).zfill(5) + 'B.jpg'
        data[filename]['voting'] = [
            row['CVM Pitipat'],
            row['CVM Prinya'],
            row['CVM Supakit'],
            row['voting'],
        ]

    return data

bans = [
    '00010S.jpg'
]

class cvmsimage():
    def __init__(self):
        self.paths = []
        path = 'Image_folder'
        folders = os.listdir(path)
        folders = sorted(folders)
        for folder in folders:
            files = [f for f in os.listdir(os.path.join(path, folder)) if f.endswith('.jpg')]
            files = sorted(files)
            for file in files:
                if file in bans:
                    continue
                self.paths += [[path, folder, file]]
        self.index = 0
        self.total = len(self.paths)
        self.modes = ['points', 'contours']
        self.mode = 'contours'
        self.windows = ['S', 'B']
        self.window = 'S'
        self.show_index = True

        self.load_image()

    def load_image(self):

        # load image
        path, folder, file = self.paths[self.index]
        if self.window == 'S':
            image_path = os.path.join(path, folder, file)
        elif self.window == 'B':
            file = file.replace('S', 'B')
            image_path = os.path.join(path, folder, folder + ' Bone window', file)
        self.image = cv2.imread(image_path)

        # change window title
        root.winfo_toplevel().title(file)

        # get roi
        self.h, self.w = self.image.shape[:2]
        x1 = int(self.w * .15)
        y1 = int(self.h * .50)
        x2 = int(self.w * .85)
        y2 = self.h
        self.roi = [x1, y1, x2, y2]

        # get points & voting
        self.label_path = os.path.join('out', file)
        self.label_path = self.label_path.replace('S.jpg', '.json')
        self.label_path = self.label_path.replace('B.jpg', '.json')
        print('self.label_path', self.label_path)
        if os.path.exists(self.label_path):
            # load data
            with open(self.label_path, 'r') as openfile:
                self.label = json.load(openfile)
        else:
            # get data
            self.make_label()

        self.show_label()
        self.draw(self.image)

    def show_label(self):
        path, folder, file = self.paths[self.index]

        print('Filename      ', file)

        print(' CVMs Pitipat ', self.label['cvms_voting'][0])
        print(' CVMs Prinya  ', self.label['cvms_voting'][1])
        print(' CVMs Supakit ', self.label['cvms_voting'][2])
        print(' Final voting ', self.label['cvms_voting'][3])

        print('     doctor     calculate        P BENZ        |    ratio (shape)     concave')
        for i, c in enumerate(['C2', 'C3', 'C4']):
            spine = self.label['spine'][i]
            l1 = shapes[spine['shape_by_doctor']]
            l2 = shapes[spine['shape_by_calculation']]
            l3 = shapes[spine['shape']]
            ratio = np.round(spine['ratio'], 2)
            auc = np.round(spine['auc'], 2)
            print('{0}   {1:13} {2:13} {3:13} | {4:13} {5:13}'.format(c, l1, l2, l3, ratio, auc))

        print('-' * 80)

    def make_label(self):

        path, folder, file = self.paths[self.index]
        points = data[file.replace('S', 'B')]['points']
        voting = data[file.replace('S', 'B')]['voting']
        self.label = {
            'filename': os.path.join(path, folder, file),
            'height': self.h,
            'width': self.w,
            'points': points,
            'points_added': [],
            'cvms_voting': voting,
            'spine': []
        }
        self.get_spine()
        for i in range(3):
            self.label['spine'][i]['shape'] = self.label['spine'][i]['shape_by_doctor']

    def get_spine(self):
 
        _contours = []
        if len(self.label['points_added']) == 2:
            _contours = [self.label['spine'][i]['contours'] for i in range(3)]
        else:
            points = self.label['points']
            c2 = [points[i] for i in [0, 1, 2, 3, 4]]
            c3 = [points[i] for i in [5, 6, 7, 8, 9, 10, 11]]
            c4 = [points[i] for i in [12, 13, 14, 15, 16, 17, 18]]
            _contours = [c2, c3, c4]

        _shapes = []
        if len(self.label['spine']) == 3:
            _shapes = [self.label['spine'][i]['shape'] for i in range(3)]            
        else:
            _shapes = [0, 0, 0]


        self.label['spine'] = []
        for i in range(3):

            contour = np.array(_contours[i])
            area = cv2.contourArea(contour)
            rectangle = cv2.boundingRect(contour)
            s = {
                'contours': _contours[i],
                'ratio': 2,
                'concave': False,
                'auc': 0,
                'area': area,
                'bbox': rectangle,
                'segmentation': [contour.flatten().tolist()],
                'shape_by_doctor': 0,
                'shape_by_calculation': 0,
                'shape': _shapes[i],
            }
            self.label['spine'].append(s)
        self.shape_by_calculation_area()
        self.shape_by_doctor()

    def shape_by_calculation_area(self):

        for i in range(3):

            contour = self.label['spine'][i]['contours']
            w1 = calc.euclidean(contour[0], contour[4])
            area = cv2.contourArea(np.array(contour[:5]))
            normal_area = area / w1
            concave = normal_area > 2

            if len(contour) == 7:
                w2 = calc.euclidean(contour[5], contour[6])
                h1 = calc.euclidean(contour[0], contour[6])
                h2 = calc.euclidean(contour[4], contour[5])
                avg_w = (w1+w2)/2
                avg_h = (h1+h2)/2
                ratio = avg_h / avg_w
            else:
                ratio = 2

            if not concave:
                self.label['spine'][i]['shape_by_calculation'] = 0
            elif ratio < 3/4:
                self.label['spine'][i]['shape_by_calculation'] = 1
            elif ratio < 4/3:
                self.label['spine'][i]['shape_by_calculation'] = 2
            else:
                self.label['spine'][i]['shape_by_calculation'] = 3

            self.label['spine'][i]['ratio'] = ratio
            self.label['spine'][i]['concave'] = concave
            self.label['spine'][i]['auc'] = normal_area

    def shape_by_doctor(self):
        v = self.label['cvms_voting'][-1]
        if v == 1:
            self.label['spine'][0]['shape_by_doctor'] = 0
            self.label['spine'][1]['shape_by_doctor'] = 0
            self.label['spine'][2]['shape_by_doctor'] = 0
        if v == 2:
            self.label['spine'][0]['shape_by_doctor'] = 3
            self.label['spine'][1]['shape_by_doctor'] = 0
            self.label['spine'][2]['shape_by_doctor'] = 0
        if v == 3:
            self.label['spine'][0]['shape_by_doctor'] = 3
            self.label['spine'][1]['shape_by_doctor'] = 1
            self.label['spine'][2]['shape_by_doctor'] = 0
        if v == 4:
            self.label['spine'][0]['shape_by_doctor'] = 3
            self.label['spine'][1]['shape_by_doctor'] = 1
            self.label['spine'][2]['shape_by_doctor'] = 1
        if v == 5:
            self.label['spine'][0]['shape_by_doctor'] = 3
            self.label['spine'][1]['shape_by_doctor'] = 2
            self.label['spine'][2]['shape_by_doctor'] = 2
        if v == 6:
            self.label['spine'][0]['shape_by_doctor'] = 3
            self.label['spine'][1]['shape_by_doctor'] = 3
            self.label['spine'][2]['shape_by_doctor'] = 3

    def save(self):
        self.get_spine()
        json_object = json.dumps(self.label, indent=4)
        with open(self.label_path, "w") as outfile:
            outfile.write(json_object)

    def update(self):
        self.get_spine()
        self.draw(self.image)
        self.show_label()
        
    def draw(self, image):

        if self.mode == 'points':
            for x, y in self.label['points']:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            for x, y in self.label['points_added']:
                cv2.circle(image, (x, y), 2, (127, 0, 255), -1)

        elif self.mode == 'contours':
            overlay = np.zeros_like(image, np.uint8)
            for i in range(3):
                spine = self.label['spine'][i]
                contour = spine['contours']
                color = colors[shapes[spine['shape']]]
                cv2.fillPoly(overlay, pts =np.array([contour]), color=color)
            image = cv2.addWeighted(image, 1, overlay, 0.3, 0)

        if self.show_index:
            for i, (x,y) in enumerate(self.label['points']):
                image = cv2.putText(image, str(i), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, .25, 
                    (0, 0, 255), 1, cv2.LINE_AA)
            for i, (x,y) in enumerate(self.label['points_added']):
                image = cv2.putText(image, str(i), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, .25, 
                    (127, 0, 255), 1, cv2.LINE_AA)

        self.display(image)

    def display(self, image):
        x1, y1, x2, y2 = self.roi
        image = image[y1:y2,x1:x2]
        self.update_tk(image)

    def update_tk(self, imagecv):

        # to tkinter image
        imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imagecv)
        imagetk = ImageTk.PhotoImage(image=im)
        panel.configure(image=imagetk)
        panel.image = imagetk

def onMouseButton(event):

    # mouse position
    x = int(event.x + cvms.roi[0])
    y = int(event.y + cvms.roi[1])
    is_changed = False

    # Left mouse button press: change class
    if event.num == 1:
        for i in range(3):
            spine = cvms.label['spine'][i]
            contour = np.array(spine['contours'])
            is_clicked = cv2.pointPolygonTest(contour, (x,y), True)
            if is_clicked > 0:
                # cvms.label['shapes'][i] = (spine['shapes']+1) % len(shapes)
                spine['shape'] = (spine['shape']+1) % len(shapes)
                is_changed = True

    # Right mouse button press: add point
    elif event.num == 2:
        if len(cvms.label['points_added']) < 2:
            cvms.label['points_added'].append([x, y])
            if len(cvms.label['points_added']) == 2:
                a, b = cvms.label['points_added']
                if a[0] > b[0]: 
                    a, b = b, a
                    cvms.label['points_added'] = [a, b]
                cvms.label['spine'][0]['contours'] += [b, a]
            is_changed = True

    if is_changed:
        cvms.update()

def onKeyPress(event):

    # Exit
    if event.keysym == 'Escape':
        root.destroy()
        exit()

    # Change mouse's marking mode
    elif event.char in ['1', '!', 'ๅ', '+']:
        cvms.window = {'S':'B', 'B':'S'}[cvms.window]
        cvms.load_image()
    elif event.char in ['2', '@', '/', '๑']:
        cvms.mode = {'points':'contours', 'contours':'points'}[cvms.mode]
        cvms.load_image()
    elif event.char in ['3', '#', '-', '๒']:
        cvms.show_index = not cvms.show_index
        cvms.load_image()

    # Change image
    elif event.char in ['a', 'A', 'ฟ', 'ฤ'] or event.keysym == 'Left':
        cvms.save()
        cvms.index = min(cvms.index - 1, cvms.total - 1)
        cvms.load_image()
    elif event.char in ['d', 'D', 'ก', 'ฏ'] or event.keysym == 'Right':
        cvms.save()
        cvms.index = min(cvms.index + 1, cvms.total - 1)
        cvms.load_image()

    elif event.char in ['z', 'Z', 'ผ', '(']:
        cvms.save()

    elif event.char in ['c', 'C', 'แ', 'ฉ']:
        cvms.make_label()
        cvms.get_spine()
        cvms.update()

if __name__ == '__main__':

    # load data points
    points_path = 'Point_FeatureExtraction'
    voting_path = 'Final_voting.csv'
    data = load_data(points_path, voting_path)

    # user inteface
    root = Tk()
    panel = tkinter.Label(root)
    panel.pack()

    cvms = cvmsimage()

    root.bind('<Button>', onMouseButton)
    root.bind('<KeyPress>', onKeyPress)
    root.mainloop() 