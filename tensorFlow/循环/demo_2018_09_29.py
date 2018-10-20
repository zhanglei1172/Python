#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os


# yuYinUtils = __import__()

wav_path = r'D:\GGdownload\data_thchs30\train'
label_file = r'D:\GGdownload\doc\doc\trans\train.word.txt'
# label

def get_waves_labels(wav_path=wav_path, label_file=label_file):
    wav_files = []
    for (dirpath, _, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('WAV'):
                filename_path = os.path.join(dirpath, filename)
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)
    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]

        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
    return new_wav_files, labels

wav_files, labels = get_waves_labels()
print(wav_files[0], labels[0])
print(wav_files.__len__(), '\n', labels.__len__())