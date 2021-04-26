#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:18:11 2021

@author: denise
"""
import parse_file
import os
import pandas as pd

# welke files
dir_path = os.path.dirname(os.path.realpath(__file__))
anno_dir = dir_path.replace("ASR2021",'Data_a/nl')
anno_files =[]
for filename in os.listdir(anno_dir):
    if filename.endswith(".plk"): 
         anno_files.append(os.path.join(anno_dir,filename))
         continue
    else:
        continue
anno_files.sort()


# hoeveel vraagtekens per file
question_count =[]
for anno in anno_files:
    count = 0
    x = parse_file.read_lines(anno)
    for y in x:
        if y[1]=='?':
            count +=1
    question_count.append([anno, count])
    
question_count = pd.DataFrame(question_count)
    