# Harris Corner Detection
Harris Corner Detector is a corner detection operator that is commonly used in computer vision algorithms to extract corners and infer features of an image. It was first introduced by Chris Harris and Mike Stephens in 1988 upon the improvement of Moravec’s corner detector. Compared to the previous one, Harris’ corner detector takes the differential of the corner score into account with reference to direction directly, instead of using shifting patches for every 45-degree angles, and has been proved to be more accurate in distinguishing between edges and corners. Since then, it has been improved and adopted in many algorithms to preprocess images for subsequent applications.

# How to execute code :
1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. You can open the folder and run following on command prompt.
> `python corner_detection.py --imagepath data/chess.png`

# Results
![output](https://github.com/Devashi-Choudhary/Corner_Detection_InImages/blob/master/Results/chess_o.png)

**Note :** For more information about harris corner detection, go through "Harris Corner Detection.pdf"
