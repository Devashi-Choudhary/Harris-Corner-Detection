import cv2
import copy
import math
import numpy as np
import argparse

def corner_points(Ix2, Iy2, Ixy):
    k=0.04
    points=[]
    points_x=[]
    points_y=[]
    for i in range(len(Ix2)):
        for j in range(len(Ix2[0])):
            c=[[0 for k in range(2)]for l in range(2)]
            c[0][0]=Ix2[i][j]
            c[0][1]=Ixy[i][j]
            c[1][0]=Ixy[i][j]
            c[1][1]=Iy2[i][j]
            det1=np.linalg.det(c)
            trace=c[0][0]+c[1][1]
            R=(det1-(k*(trace*trace)))
            if (R>0.0001):
                points.append((i,j)) 
    return points

def gradient_square(X_gradient, Y_gradient, gaussian_mask, n, padding):
    I_xx=X_gradient*X_gradient
    I_yy=Y_gradient*Y_gradient
    I_xy=X_gradient*Y_gradient
    avg=0
    Ix2=copy.deepcopy(I_xx)
    Iy2=copy.deepcopy(I_yy)
    Ixy=copy.deepcopy(I_xy)
    for i in range(n):
            for j in range(n):
                avg=avg+gaussian_mask[i][j]
    for i in range(padding,len(Ix2)-padding):
        for j in range(padding,len(Ix2[0])-padding):
            sx=0
            sy=0
            sxy=0
            for k in range(len(gaussian_mask)):
                for l in range(len(gaussian_mask)):
                    x=abs(k-i-padding)
                    y=abs(l-j-padding)
                    sx=sx+(gaussian_mask[k][l]*I_xx[x][y])
                    sy=sy+(gaussian_mask[k][l]*I_yy[x][y])
                    sxy=sxy+(gaussian_mask[k][l]*I_xy[x][y])
            sx=sx/avg
            sy=sy/avg
            sxy=sxy/avg
            Ix2[i][j]=sx
            Iy2[i][j]=sy
            Ixy[i][j]=sxy
    Ix2=Ix2[padding:-padding, padding:-padding]
    Iy2=Iy2[padding:-padding, padding:-padding]
    Ixy=Ixy[padding:-padding, padding:-padding]
    Ix2 = Ix2/np.max(Ix2)
    Iy2 = Iy2/np.max(Iy2)
    Ixy = Ixy/np.max(Ixy)
    return Ix2, Iy2, Ixy

def create_gaussian_mask(n):
    x = np.arange(-math.floor(n/2), math.floor((n+1)/2))
    y = np.arange(-math.floor(n/2), math.floor((n+1)/2))
    a,b= np.meshgrid(x,y);
    gaussian_mask=[[0 for i in range(n)]for j in range(n)]
    sigma=1.5
    for i in range(n):
        for j in range(n):
            p=a[i][j]
            q=b[i][j]
            c=(1/(2*3.14*sigma*sigma))
            d=-((p*p)+(q*q))/(2*sigma*sigma)
            gaussian_mask[i][j]=c*math.exp(d)
    return gaussian_mask

def gradient(gray_scale_image, sobel_x, sobel_y):
    X_gradient=copy.deepcopy(gray_scale_image)
    Y_gradient=copy.deepcopy(gray_scale_image)
    for i in range(1,len(gray_scale_image)-1):
        for j in range(1,len(gray_scale_image[0])-1):
            sx=0
            sy=0
            for k in range(len(sobel_x)):
                for l in range(len(sobel_x)):
                    x=abs(k-i-1)
                    y=abs(l-j-1)
                    sx=sx+(sobel_x[k][l]*gray_scale_image[x][y])
                    sy=sy+(sobel_y[k][l]*gray_scale_image[x][y])
            X_gradient[i][j]=sx
            Y_gradient[i][j]=sy
    for i in range(len(X_gradient)):
        for j in range(len(X_gradient[0])):
            if X_gradient[i][j]<0:
                X_gradient[i][j]=X_gradient[i][j]*(-1)
    for i in range(len(X_gradient)):
        for j in range(len(X_gradient[0])):
            if X_gradient[i][j]>255:
                X_gradient[i][j]=255
    x=X_gradient.astype(np.uint8)
#     cv2.imshow("X_gradient",x)
#     cv2.waitKey(0)
   
    for i in range(len(Y_gradient)):
        for j in range(len(Y_gradient[0])):
            if Y_gradient[i][j]<0:
                Y_gradient[i][j]=Y_gradient[i][j]*(-1)
    for i in range(len(Y_gradient)):
        for j in range(len(Y_gradient[0])):
            if Y_gradient[i][j]>255:
                Y_gradient[i][j]=255
    y=Y_gradient.astype(np.uint8)
#     cv2.imshow("Y_gradient",y)
#     cv2.waitKey(0)
    
    return X_gradient, Y_gradient

def DownSample(image):
    drdc1=image[::2,::2]
    return drdc1

def padded_image(gray_image, padding):
    gray_scale_image=np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding]=gray_image[i][j] 
    return gray_scale_image

def corner_detection(imagepath):
    n=3
    padding=math.ceil(n/2)-1
    sobel_x=[[-1,-2,-1],[0,0,0],[1,2,1]]
    sobel_y=[[-1,0,1],[-2,0,2],[-1,0,1]]
    gray_image=cv2.imread(imagepath,0)
    #gray_image=DownSample(gray_image1)
    gray_scale_image = padded_image(gray_image, padding)
    X_gradient, Y_gradient = gradient(gray_scale_image, sobel_x, sobel_y)
    gaussian_mask = create_gaussian_mask(n)
    Ix2, Iy2, Ixy = gradient_square(X_gradient, Y_gradient, gaussian_mask, n, padding)
    points = corner_points(Ix2, Iy2, Ixy)
    corner=copy.deepcopy(gray_image)
    for i,j in points:
        corner[i][j]=0
    corner=corner.astype(np.uint8)
    cv2.imshow("Corner_Detection",corner)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagepath", required = True, help = "path to input image")
args = ap.parse_args()

if __name__ == "__main__":
    imagepath = args.imagepath
    corner_detection(imagepath)

