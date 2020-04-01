import numpy as np
from math import sqrt


def getKEspectrum(field):

    u = field[0]
    v = field[1]
    eps = 1e-50 # to void log(0)


    amplsU = abs(np.fft.fftn(u)/u.size)
    amplsV = abs(np.fft.fftn(v)/v.size)

    EK_U  = amplsU**2
    EK_V  = amplsV**2

    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]

    box_sidex = sign_sizex
    box_sidey = sign_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]
            EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j]

    EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr)

    return EK_avsphr
