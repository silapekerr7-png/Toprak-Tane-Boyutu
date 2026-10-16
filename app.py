import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pickle
from ozellik import ozellik_cikar, etkin_ppm
 
st.title("Toprak Tane Boyutu Tahmini :mountain:")
model=pickle.load(open('toprak.pkl','rb'))
ppm=pd.read_csv('ppm_updated.csv')
x=np.array([0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200])
 
kamera=st.selectbox('Fotoğrafı çeken kamera',ppm['phone'])
dosyalar=st.file_uploader('Aynı numunenin fotoğraflarını yükle',type=['jpg','jpeg','png'],accept_multiple_files=True)
 
if dosyalar and st.button('Tahmin et'):
    satir=ppm[ppm['phone']==kamera].iloc[0]
    ozellikler=[]
    for d in dosyalar:
        img=cv2.imdecode(np.frombuffer(d.read(),np.uint8),1)
        H,W=img.shape[:2]
        p=etkin_ppm(satir['ppm'],satir['width'],satir['height'],W,H)
        ozellikler.append(ozellik_cikar(img,p))
    tahmin=model.predict([np.mean(ozellikler,axis=0)])[0]
    tahmin=np.maximum.accumulate(np.clip(tahmin,0,100))
    tahmin[-1]=100
    d50=10**np.interp(50,tahmin,np.log10(x))
    st.success(f'Tahmini medyan tane çapı (d50): {d50:.2f} mm')
    st.line_chart(pd.DataFrame({'log10 çap (mm)':np.log10(x),'geçen %':tahmin}).set_index('log10 çap (mm)'))
    st.write(pd.DataFrame({'çap (mm)':x,'geçen %':tahmin.round(1)}))
 