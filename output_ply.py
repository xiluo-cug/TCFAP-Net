from tool import Plot
import numpy as np
from helper_ply import read_ply
import os

label_path = "/home/kukdo/data/SensatUrban-DLA_Glocal_AttRes/test/Log_2023-06-07_06-19-04/test_preds" # pred label path
dir_list = os.listdir(label_path)
data_path = "/home/kukdo/Workspace/SensatUrban/Dataset/SensatUrban/original_block_ply" # data path
filename = []
save_path = "/home/kukdo/data/SensatUrban-DLA_Glocal_AttRes/output/20230607" # ply save path

for i in range(len(dir_list)):
    if dir_list[i][-5:] == 'label':
        filename += [dir_list[i][:-6]]
# label to rgb
ins_colors = [[85, 107, 47],  # ground -> OliveDrab
              [0, 255, 0],  # tree -> Green
              [255, 165, 0],  # building -> orange
              [41, 49, 101],  # Walls ->  darkblue
              [0, 0, 0],  # Bridge -> black
              [0, 0, 255],  # parking -> blue
              [255, 0, 255],  # rail -> Magenta
              [200, 200, 200],  # traffic Roads ->  grey
              [89, 47, 95],  # Street Furniture  ->  DimGray
              [255, 0, 0],  # cars -> red
              [255, 255, 0],  # Footpath  ->  deeppink
              [0, 255, 255],  # bikes -> cyan
              [0, 191, 255]  # water ->  skyblue
             ]

for i in range(len(filename)):
    print("num = ",i)
    path = os.path.join(data_path,filename[i]+'.ply')
    pre_path = os.path.join(label_path,filename[i]+'.label')
    raw_data = read_ply(path)

    pc = np.vstack((raw_data['x'], raw_data['y'], raw_data['z'])).T
    raw_color = np.vstack((raw_data['red'], raw_data['green'], raw_data['blue'])).T
    pre_label = np.fromfile(pre_path,np.uint8)
    pre_colors = np.zeros((pre_label.shape[0], 3))
    sem_ins_labels = np.unique(pre_label)

    for id, semins in enumerate(sem_ins_labels):
        valid_ind = np.argwhere(pre_label == semins)[:, 0]
        if semins <= -1:
            tp = [0, 0, 0]
        else:
                tp = ins_colors[id]

        pre_colors[valid_ind] = tp

    pre_data = np.hstack((pc,pre_colors))

    Plot.save_ply_o3d(pre_data,os.path.join(save_path,filename[i]+".ply"))

