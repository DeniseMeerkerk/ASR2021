#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:29:59 2021

@author: denise
"""

#%%Code snippets (python)
# Read wav and sample frequency:
import librosa
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = dir_path.replace("ASR2021",'Data/CGN_comp_ab/vol/bigdata2/corpora2/CGN2/data/audio/wav/comp-b/nl')

wav_files =[]
for filename in os.listdir(data_dir):
    if filename.endswith(".wav"): 
         wav_files.append(os.path.join(data_dir,filename))
         continue
    else:
        continue
#%%
X, sample_rate = librosa.load(<file name>, sr=None, offset=0)
#Create spectrogram:
spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, hop_length=sample_rate/100)
#Create MFCC matrix:
#https://librosa.org/doc/main/generated/librosa.feature.mfcc.html


# https://github.com/wilrop/Import-CGN

"""
De data

Annotaties van de interviews met leraren hebben niet allemaal de interview 
vragen. "plk" bijvoorbeeld wel. Heb alleen het idee dat het niet helemaal 
compleet is.

Voorbeeld:
<au id="1" s="N00059" tb="0.000">
ja	TSW()	ja	141336	45366	
waarom	BW()	waarom	620539	135546	
bent	WW(pv,tgw,met-t)	zijn	400380	122511	
u	VNW(pers,pron,nomin,vol,2b,getal)	u	620394	135431	
uh	TSW()	uh	381783	125605	
leraar	N(soort,ev,basis,zijd,stan)	leraar	172677	56550	
geworden	WW(vd,vrij,zonder)	worden	109930	120427	
?	LET()	?	0	0			
<au id="2" s="N00060" tb="3.122">
dat	VNW(aanw,pron,stan,vol,3o,ev)	dat	619602	134794	
is	WW(pv,tgw,ev)	zijn	141101	122511	
een	LID(onbep,stan,agr)	een	619677	134836	
uh	TSW()	uh	381783	125605	
opmerking	N(soort,ev,basis,zijd,stan)	opmerking	221403	72624	
van	VZ(init)	van	620421	135442	
een	LID(onbep,stan,agr)	een	619677	134836	
leraar	N(soort,ev,basis,zijd,stan)	leraar	172677	56550	
bij	VZ(init)	bij	619448	134687	
mij	VNW(pr,pron,obl,vol,1,ev)	mij	620098	135181	
op	VZ(init)	op	620227	135275	
de	LID(bep,stan,rest)	de	619612	134796	
MAVO	N(eigen,ev,basis,zijd,stan)	MAVO	184750	61090	
geweest	WW(vd,vrij,zonder)	zijn	109500	122511	
van	VZ(init)	van	620421	135442	
als	VG(onder)	als	619409	134654	
je	VNW(pers,pron,nomin,red,2v,ev)	je	620014	135108	
niks	VNW(onbep,pron,stan,vol,3o,ev)	niks	620157	135222	
kan	WW(pv,tgw,ev)	kunnen	145430	47033	
worden	WW(inf,vrij,zonder)	worden	370616	120427	
dan	BW()	dan	619596	134792	
kan	WW(pv,tgw,ev)	kunnen	145430	47033	
je	VNW(pers,pron,nomin,red,2v,ev)	je	620014	135108	
altijd	BW()	altijd	619419	134663	
nog	BW()	nog	620161	135226	
in	VZ(init)	in	620000	135095	
't	LID(bep,stan,evon)	het	619904	135669	
onderwijs	N(soort,ev,basis,onz,stan)	onderwijs	209903	69663	

"""