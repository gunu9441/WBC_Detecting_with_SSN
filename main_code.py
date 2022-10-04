#!/usr/bin/env python
# coding: utf-8

# In[1]:
# include 단계

import shutil
from PIL import Image, ImageDraw
import imageio
import seaborn as sns
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation
from dv import AedatFile
import random
import pandas as pd
from learningStats import learningStats
import slayerSNN as snn
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")
#from datetime import datetime


# In[2]:

# DBSCAN 클러스터링 클래스

# 파일 하나당 한 클래스 사용
class clustering:
    def __init__(self, frame_size=40000):
        self.frame_size = frame_size
        self.events = None
        self.df = None
        self.start_time = 0

    # aedat4파일 읽어서 이벤트로 메모리에 저장
    def read_aedat_file(self, path):
        print()
        with AedatFile(path) as f:
            events = np.hstack([packet for packet in f['events'].numpy()])
            timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
            self.events = events
            self.timestamp_zeroization()
            self.dflization()
            return events

    # read_aedat_file에서 호출
    # timestamp 0부터 시작하게
    def timestamp_zeroization(self):
        temp = self.events['timestamp'][0]
        self.events['timestamp'] -= temp

    # read_aedat_file에서 호출
    # df 초기화
    def dflization(self):
        self.df = pd.DataFrame(self.events)

    # 시작시간부터 원하는 프레임으로 자르기
    def cut_frame(self, start_time=0):
        end_condition = self. df['timestamp'] <= start_time+self.frame_size
        start_condition = self.df['timestamp'] >= start_time
        #print('end_condition : {}'.format(end_condition))
        #print('start_condition : {}'.format(start_condition))
        target_df = self.df[end_condition & start_condition]
        a = target_df.reset_index(drop=True)
        a = self.time_nomalization(a)
        return a

    # cyt_frame에서 호출
    # 시간축이 스케일이 너무 커서 스케일 맞춰줌
    def time_nomalization(self, frame):
        temp = self.frame_size/150
        # cpy=pd.DataFrame.copy(frame)
#         for i in range(len(frame)):
#             frame['timestamp'][i] = frame['timestamp'][i]/temp
        frame['timestamp'] /= temp
        return frame

    # 실제 DBSCAN으로 클러스터링해서 결과 반환
    def make_model(self, feature, eps=17, min_samples=100):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        predict = pd.DataFrame(model.fit_predict(feature))
        predict.columns = ['predict']
        # concatenate labels to df as a new column
        r = pd.concat([feature, predict], axis=1)
        return r


# 그리기
def draw(r):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(r['x'], r['y'], r['timestamp'], c=r['predict'], alpha=0.5)
    ax.view_init(270, 270)
    ax.set_xlabel('Sepal lenth')
    ax.set_ylabel('Sepal width')
    ax.set_zlabel('Petal length')

    plt.show()
    return r

# 파일로 저장


def save(r, i):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(r['x'], r['y'], r['timestamp'], c=r['predict'], alpha=0.5)
    ax.view_init(270, 270)
    ax.set_xlabel('Sepal lenth')
    ax.set_ylabel('Sepal width')
    ax.set_zlabel('Petal length')
    plt.savefig("./save/dv5/test{}.png".format(i))


# In[3]:


# 판다스 배열들의 리스트 만들기
def event_output(p1):
    cname = ['timestamp', 'x', 'y', 'polarity', '_p1', '_p2', 'predict']
    r_list = []

    # 검출한거 개수만큼 미리 리스트 만들어놓기
    r_list_size = p1['predict'].max(axis=0)
    for j in range(r_list_size+1):
        newp = pd.DataFrame(columns=cname)
        r_list.append(newp)

    for i in p1.index:
        # print(p1.loc[i].values)
        s = p1.loc[i].values.tolist()
        if p1.loc[i, "predict"] == -1:
            continue
        r_list[p1.loc[i, "predict"]].loc[len(r_list[p1.loc[i, "predict"]])] = s

    # print(r_list)
    return r_list

# csv파일로 출력


def csv_output(frame, num, r_list, to_hy):
    # num은 출력될 파일이름(전체에서 파일 순서)
    # r_list는 출력할 df 리스트
    list_size = len(r_list)

    for i in range(0, list_size):
        print("num : "+str(num+i))
        df = r_list[i]
        # draw(df)
        # save(df,num+i)
        df.to_csv("./model/dv_dataset/"+str(num+i)+".csv")
        '''여기서 저장할 파일 위치'''
        to_hy.loc[num+i] = [frame, num+i, df['x'].max(), df['y'].max(),
                            df['x'].min(), df['y'].min()]

    # 반환값은 새로운 num
    return num+list_size


# In[4]:

# slayer 사용하기 위해서 dv 데이터셋 받아오는 클래스
# 이벤트를 받아와서 슬레이어에 넣을수 있게 가공
class dvDataset(Dataset):
    def __init__(self, datasetPath, samplingTime, sampleLength, pix):
        self.path = datasetPath
        self.csv_names = []

        for csv_candidate in os.listdir(datasetPath):
            if '.csv' in csv_candidate:
                self.csv_names.append(int(csv_candidate.rstrip('.csv')))
        self.csv_names.sort()
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)
        self.pix = pix

    def __getitem__(self, index):
        # load dataframes what we are going to use.
        use_col = ['x', 'y', 'polarity', 'timestamp']
        df = pd.read_csv(
            self.path + str(self.csv_names[index])+'.csv', usecols=use_col)

        p_xEvent = df['x']
        p_yEvent = df['y']
        p_pEvent = df['polarity']
        p_tEvent = df['timestamp']

        x_min = p_xEvent.min()
        y_min = p_yEvent.min()

        p_xEvent = p_xEvent-x_min
        p_yEvent = p_yEvent-y_min

        p_xEvent = (p_xEvent / x_min * self.pix)
        p_yEvent = (p_yEvent / y_min * self.pix)
        p_tEvent = p_tEvent-p_tEvent.min()

        xEvent = p_xEvent.tolist()
        yEvent = p_yEvent.tolist()
        pEvent = p_pEvent.tolist()
        tEvent = p_tEvent.tolist()

        inputSpikes = snn.io.event(xEvent, yEvent, pEvent, tEvent).toSpikeTensor(
            torch.zeros((2, self.pix, self.pix, self.nTimeBins)), samplingTime=self.samplingTime
        )

        return inputSpikes

    def __len__(self):
        return len(self.csv_names)


# In[5]:

# 슬레이어 네트워크
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 16, 5, padding=1)
        self.conv2 = slayer.conv(16, 16, 3, padding=1)
        self.pool1 = slayer.pool(2)

        self.conv3 = slayer.conv(16, 32, 3, padding=1)
        self.conv4 = slayer.conv(32, 32, 3, padding=1)
        self.pool2 = slayer.pool(2)

        self.conv5 = slayer.conv(32, 64, 3, padding=1)
        self.conv6 = slayer.conv(64, 64, 3, padding=1)
        self.conv7 = slayer.conv(64, 64, 3, padding=1)
        self.conv8 = slayer.conv(64, 64, 3, padding=1)
        self.pool3 = slayer.pool(2)

        '''
        self.conv9 = slayer.conv(64, 128, 3, padding=1)
        self.conv10 = slayer.conv(128, 128, 3, padding=1)
        self.conv11 = slayer.conv(128, 128, 3, padding=1)
        self.conv12 = slayer.conv(128, 128, 3, padding=1)
        self.pool4 = slayer.pool(2)
        '''

        self.fc1 = slayer.dense((4, 4, 64), 2)

        '''
        self.fc2   = slayer.dense((4, 4, 128), 16)
        self.fc3   = slayer.dense((2, 2, 16), 2)
        '''

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(
            self.slayer.psp(spikeInput)))  # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.conv2(
            self.slayer.psp(spikeLayer1)))  # 32, 32, 16
        spikePool1 = self.slayer.spike(self.pool1(
            self.slayer.psp(spikeLayer2)))  # 16, 16, 16

        spikeLayer3 = self.slayer.spike(self.conv3(
            self.slayer.psp(spikePool1)))  # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.conv4(
            self.slayer.psp(spikeLayer3)))  # 16, 16, 32
        spikePool2 = self.slayer.spike(self.pool2(
            self.slayer.psp(spikeLayer4)))  # 8,  8, 32

        spikeLayer5 = self.slayer.spike(self.conv5(
            self.slayer.psp(spikePool2)))  # 8,  8, 64
        spikeLayer6 = self.slayer.spike(self.conv6(
            self.slayer.psp(spikeLayer5)))  # 8,  8, 64
        spikeLayer7 = self.slayer.spike(self.conv7(
            self.slayer.psp(spikeLayer6)))  # 8,  8, 64
        spikeLayer8 = self.slayer.spike(self.conv8(
            self.slayer.psp(spikeLayer7)))  # 8,  8, 64
        spikePool3 = self.slayer.spike(self.pool3(
            self.slayer.psp(spikeLayer8)))  # 4,  4, 64
        #print('spikeLayer6 size:', spikeLayer6.size())
        # spikeFC1    = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer6))) #  128
        #print('spikeFC1 size:', spikeFC1.size())
        # spikeFC2    = self.slayer.spike(self.fc2  (self.slayer.psp(spikeFC1))) #  16
        #print('spikeFC2 size:', spikeFC2.size())
        spikeOut = self.slayer.spike(
            self.fc1(self.slayer.psp(spikePool3)))  # 2
        '''
        print("spikeInput : ", spikeInput.size())
        print("spikeLayer1 : ", spikeLayer1.size())
        print("spikeLayer2 : ", spikeLayer2.size())
        print("spikePool1 : ", spikePool1.size())
        print("spikeLayer3 : ", spikeLayer3.size())
        print("spikeLayer4 : ", spikeLayer4.size())
        print("spikePool2 : ", spikePool2.size())  
        print("spikeLayer5 : ", spikeLayer5.size()) 
        print("spikeLayer6 : ", spikeLayer6.size())
        print("spikeLayer7 : ", spikeLayer7.size())
        print("spikeLayer8 : ", spikeLayer8.size())
        print("spikePool3 : ", spikePool3.size())
        print('spikeOut size:', spikeOut.size())
        '''

        return spikeOut


# In[6]:

# 슬레이어로 예측하는 함수
# Prediction
def prediction(testLoader, model, device, net):
    batch_size = testLoader.batch_size
    prediction_list = [None]
    with torch.no_grad():
        for i, input in enumerate(testLoader, 0):
            input = input.to(device)
            output = net.forward(input)

            predictions = snn.predict.getClass(output).cpu().detach().numpy()

            for prediction in predictions:
                prediction_list.append(prediction)

        return prediction_list


# In[7]:

# 시각화 클래스
class visualization:
    def __init__(self, frame=25):  # 하영 프레임 수정 50 -> 25
        self.aedat_file = None
        self.aedat_df = None
        self.frame = frame
        self.timestamp_unit = 1000 / frame

        self.height = None
        self.width = None

    # 생 aedat4파일 읽는 함수(사용x)
    def load_aedat4(self, aedat_file, verbose=False):
        """
        load_aedat4():
            This function can load a aedat4.0 file and generate a dataframe based on the
            aedat4.0 file.
                aedat_file  : The path where aedat file exists.
                verbose     : You can see the particular values of meaningful variables
                              if verbose is True.
        """
        self.aedat_file = aedat_file
        xEvent = list()
        yEvent = list()
        pEvent = list()
        tEvent = list()
        if verbose:
            print("-----        READING Aedat4.0 PROCESS        -----")
        with AedatFile(self.aedat_file) as f:
            for idx, event in enumerate(f["events"]):
                xEvent.append(event.x)
                yEvent.append(event.y)
                pEvent.append(event.polarity)
                # based on 1ms timestamp.
                tEvent.append(int(round(event.timestamp / 1000)))
                if verbose:
                    if idx % 10000 == 0:
                        print("PROGRESS: {:11d}".format(idx))
        self.aedat_df = pd.DataFrame(
            {"x": xEvent, "y": yEvent, "p": pEvent, "t": tEvent}
        )
        self.height = self.aedat_df["y"].max() + 1
        self.width = self.aedat_df["x"].max() + 1
        self.aedat_df["t"] = self.aedat_df["t"] - self.aedat_df["t"].min()
        if verbose:
            print("-----    Complete Mapping list -> DataFrame    -----")
            print(
                "xEvent    min: {}, max: {}".format(
                    self.aedat_df["x"].min(), self.aedat_df["x"].max()
                )
            )
            print(
                "yEvent    min: {}, max: {}".format(
                    self.aedat_df["y"].min(), self.aedat_df["y"].max()
                )
            )
            print(
                "timestamp min: {}, max: {}".format(
                    self.aedat_df["t"].min(), self.aedat_df["t"].max()
                )
            )
            print("width        : {}".format(self.width))
            print("height       : {}".format(self.height))

    # pkl로 저장된 파일 읽기(사용x)
    def load_pkl(self, path, verbose=False):
        """
        load_pkl():
            This function can load a pickle file and generate a dataframe based on
            the pickle.
                path        : a path where you want to load a pickle file.
        """
        self.aedat_df = pd.read_pickle(path)
        self.height = self.aedat_df["y"].max() + 1
        self.width = self.aedat_df["x"].max() + 1
        self.aedat_df["t"] = self.aedat_df["t"] - self.aedat_df["t"].min()
        if verbose:
            print("-----    Complete Mapping pickle -> DataFrame    -----")
            print(
                "xEvent    min: {}, max: {}".format(
                    self.aedat_df["x"].min(), self.aedat_df["x"].max()
                )
            )
            print(
                "yEvent    min: {}, max: {}".format(
                    self.aedat_df["y"].min(), self.aedat_df["y"].max()
                )
            )
            print(
                "timestamp min: {}, max: {}".format(
                    self.aedat_df["t"].min(), self.aedat_df["t"].max()
                )
            )
            print("width        : {}".format(self.width))
            print("height       : {}".format(self.height))
    # pkl로 저장(사용x)

    def save_df_pkl(self, path, verbose=False):
        """
        save_df_pkl():
            This function can save the dataframe to pickle.
                path        : a path where you want to save a pickle file.
        """
        self.aedat_df.to_pickle(path)
        if verbose:
            print("Saved the dataframe to pickle in {}!".format(path))
    # 2차원 박스로 테두리 그리기

    def draw_2D_bbox(self, label_list, info_list, File_name, folder="origin", verbose=False):
        """
        draw_2D_bbox():
            This function gets aedat4.0 file and visualize it with bounding box
            based on PIL.
            The background color is gray.
            arguments:
                start_index     : You can select 1 frame by selecting index in
                                  range from 0 to any index you want to
                                  visualize.
                min_point       : You can make the bounding box with these two
                                  arguments.
                                  In the picture, (0,0) points is located on the left
                                  upside the cornor.
                max_point       : It is similar to min_point argument. This argument
                                  is maximum point in four coordinates which mean
                                  bounding box.
                verbose         : You can see the particular values of meaningful
                                  variables if verbose is True.
        """
        checkbox = [175, 175, 225, 259]
        images = []
        wbc_count = 0
        frame_count = 0
        for i in range(0, len(info_list)):
            frame, _, x_max, y_max, x_min, y_min = info_list.iloc[i]
            # condition
            if not(225 < x_min or 175 > x_max or 175 > y_max or 259 < y_min):
                frame_count += 1
                if frame_count > 6:
                    wbc_count += 1
                    frame_count = 0
            # condition end
            if i == 0:
                before_frame = -1
            start_timestamp = frame * self.timestamp_unit
            end_timestamp = (frame + 1) * self.timestamp_unit - 1
            """if verbose:
                print("----------------------------------------")
                print("index:                  {}".format(start_index))
                print("limited_timestamp_unit: {}".format(self.timestamp_unit))
                print("start_timestamp:        {}".format(start_timestamp))
                print("end_timestamp:          {}".format(end_timestamp))"""

            end_condition = self.aedat_df["t"] <= end_timestamp
            start_condition = self.aedat_df["t"] >= start_timestamp

            # Generate preprocessed dataframe
            target_df = self.aedat_df[end_condition & start_condition]

            image = Image.new(
                mode="RGB", size=(self.width, self.height), color=(128, 128, 128)
            )
            if verbose:
                print("-----            Save Stage        -----")
            for _, row in target_df.iterrows():
                image.putpixel(
                    (row["x"], row["y"]),
                    (int(row["p"]) * 255, int(row["p"])
                     * 255, int(row["p"] * 255)),
                )

            if label_list[i+1] == 1:
                # Draw a Bounding box
                min_point = (x_min, y_min)  # 좌표값
                max_point = (x_max, y_max)  # 좌표값
                if i == 0 or before_frame != frame:
                    bbox_image = ImageDraw.Draw(image)
                    bbox_image.rectangle(
                        (min_point, max_point), outline=(255, 0, 0), width=3)
                    bbox_image.rectangle(
                        ((175, 175), (225, 259)), outline=(0, 0, 0), width=3)
                    images.append(image)
                elif before_frame == frame:
                    bbox_image = ImageDraw.Draw(images[-1])
                    bbox_image.rectangle(
                        (min_point, max_point), outline=(255, 0, 0), width=3)
                    bbox_image.rectangle(
                        ((175, 175), (225, 259)), outline=(0, 0, 0), width=3)
#                 bbox_image.text(20,20,str(wbc_count))
#                 bbox_image.text(20,40,str(frame_count))

                before_frame = frame

            # images.append(image)

        """image.save(
            "../output/2D_images/{}/test{}_bbox.png".format(folder, start_index + 1),
            "png",
        )"""
        imageio.mimsave('./model/output/'+File_name+'.gif', images, fps=2.0)
        return wbc_count


# In[8]:

# os관련 함수
# 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
# 전체 파일 삭제


def DeleteAllFiles(directory):
    if os.path.exists(directory):
        for file in os.scandir(directory):
            os.remove(file.path)
        return "Remove done"
    else:
        return "dir not found"


# In[9]:

# 메인 함수
def init(File_name, con=1):
    test2 = clustering()
    # 앱 개발파트에서 파라미터로 파일이름 넘겨줍니다.
    '''여기서 불러올 파일제목 서버에서 받아온 이름 파라미터로 넣어주기'''
    test2.read_aedat_file('./model/input/'+File_name+'.aedat4')
    if con == 1:
        createFolder('./model/dv_dataset')
    num = 1
    to_hy = pd.DataFrame(
        columns=['frame', 'num', 'x_max', 'y_max', 'x_min', 'y_min'])
    for i in range(1, 10000):  # 돌려보고 끝나는 숫자 설정
        try:
            size = i*40000
            sg1 = test2.cut_frame(size)
            sg2 = test2.make_model(sg1)
            # save(sg2,num)
            sg3 = event_output(sg2)
            num = csv_output(i, num, sg3, to_hy)
        except:
            break

    netParams = snn.params('./model/demo.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: {}'.format(device))

    random_seed = 777
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    net = Network(netParams).to(device)

    testingSet = dvDataset(datasetPath=netParams['training']['path']['in'],
                           samplingTime=netParams['simulation']['Ts'],
                           sampleLength=netParams['simulation']['tSample'],
                           pix=netParams['simulation']['pixel'])
    testLoader = DataLoader(
        dataset=testingSet, batch_size=2, shuffle=False, num_workers=4)

    net.load_state_dict(torch.load('./model/fuct.pt'))

    prediction1 = prediction(testLoader, net, device, net)
    print('모든 데이터들의 백혈구 판단 결과:', prediction1)
    print('전체 데이터 개수: ', len(prediction1))

    print("?")

    origin_instance = visualization(frame=25)
    origin_instance.load_aedat4(
        "./model/input/"+File_name+".aedat4", verbose=True
    )
    #origin_instance.save_df_pkl("./1.pkl", verbose=True)

#     preprocessed_instance = visualization(frame=50)

    """preprocessed_instance.load_aedat4(
        "./newdata.aedat4", verbose=True
    )"""
    """preprocessed_instance.save_df_pkl(
        "./1.pp.pkl", verbose=True
    )
"""
#     origin_instance.load_pkl("./1.pkl", verbose=True)
    """preprocessed_instance.load_pkl(
        "./1.pp.pkl", verbose=True
    )"""

    #origin_instance.draw_2D(folder="origin", start_index=6, sequence=1000, verbose=True)

    count = origin_instance.draw_2D_bbox(
        label_list=prediction1,
        info_list=to_hy,
        folder="origin",
        verbose=False,
        File_name=File_name
    )
    print("개수 : ", count)
    if con == 1:
        shutil.rmtree('./model/dv_dataset')
    return count


init("test")

# In[10]:


#count = init('bbb')


# In[11]:


# image = Image.new(
#                 mode="RGB", size=(346, 260), color=(128, 128, 128)
#             )
# min_point=(200, 200) # 좌표값
# max_point=(260, 260) # 좌표값
# bbox_image = ImageDraw.Draw(image)
# bbox_image.rectangle((min_point, max_point), outline=(255, 0, 0), width=3)
# image
