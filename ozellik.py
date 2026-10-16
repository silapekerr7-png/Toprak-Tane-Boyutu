import cv2
import numpy as np

ISIK_PPM=4      # bantlar için çalışma çözünürlüğü (piksel/mm)
INCE_PPM=8      # ince doku için çalışma çözünürlüğü
SIGMA_MM=[0.25,0.5,1,2,4,8,16,32]   # Gaussian ölçekleri (mm)
RADYUS_MM=[1,2,4,8,16]              # morfolojik açma/kapama yarıçapları (mm)

def ozellik_isimleri():
    isim=['gri_ort','gri_std','r_ort','g_ort','b_ort','doygunluk','ince_laplace','ince_gradyan']
    isim+=[f'bant_{i}' for i in range(len(SIGMA_MM)-1)]
    isim+=['bant_toplam']
    isim+=[f'acma_{r}' for r in RADYUS_MM]
    isim+=[f'kapama_{r}' for r in RADYUS_MM]
    return isim

def etkin_ppm(tablo_ppm,tablo_w,tablo_h,img_w,img_h):
    # fotoğraf tablodaki çözünürlükten küçültülmüşse ppm'i orantılı küçült
    return tablo_ppm*max(img_w,img_h)/max(tablo_w,tablo_h)

def yeniden_boyutlandir(gri,ppm,hedef_ppm):
    oran=hedef_ppm/ppm
    return cv2.resize(gri,None,fx=oran,fy=oran,interpolation=cv2.INTER_AREA)

def ozellik_cikar(bgr,ppm):
    # bgr: cv2 ile okunmuş renkli fotoğraf, ppm: bu fotoğraftaki piksel/mm
    gri=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    f=[gri.mean(),gri.std(),bgr[:,:,2].mean(),bgr[:,:,1].mean(),bgr[:,:,0].mean(),hsv[:,:,1].mean()]

    # ince doku: 8 px/mm çözünürlükte ortadaki 800x800 kare
    ince=yeniden_boyutlandir(gri,ppm,INCE_PPM).astype(np.float32)/255
    h,w=ince.shape; k=min(400,h//2,w//2)
    kare=ince[h//2-k:h//2+k,w//2-k:w//2+k]
    f.append(np.log(cv2.Laplacian(kare,cv2.CV_32F).var()+1e-9))
    f.append(np.abs(cv2.Sobel(kare,cv2.CV_32F,1,0)).mean())

    # bantlar: Gaussian farkı enerjileri (4 px/mm)
    g=yeniden_boyutlandir(gri,ppm,ISIK_PPM).astype(np.float32)/255
    bulanik=[g]+[cv2.GaussianBlur(g,(0,0),s*ISIK_PPM) for s in SIGMA_MM]
    enerji=np.array([np.mean((bulanik[i]-bulanik[i+1])**2) for i in range(len(SIGMA_MM))])[:-1]
    f+=list(enerji/enerji.sum())
    f.append(np.log(enerji.sum()+1e-12))

    # granülometri: açma (parlak taneler) ve kapama (koyu taneler) (2 px/mm)
    g2=yeniden_boyutlandir(gri,ppm,2)
    for islem in [cv2.MORPH_OPEN,cv2.MORPH_CLOSE]:
        for r in RADYUS_MM:
            r_px=max(1,int(round(r*2)))
            ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r_px+1,2*r_px+1))
            f.append(cv2.morphologyEx(g2,islem,ker).mean()/(g2.mean()+1e-9))
    return np.array(f,dtype=float)
