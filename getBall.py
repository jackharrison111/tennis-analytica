
import numpy as np
from SourceCode.VideoReader import VideoReader
from SourceCode.FindBall import BallFinder
import cv2
import matplotlib.pyplot as plt
import copy as cp


filename = "StereoClips/stereoClip1_Kyle.mov"

filename2 = "HomeClips/Dad/behindFence/middleRally2_Dad.mp4"
filename3 = "HomeClips/edited/Dad/rally1-edit_Dad.mp4"

vr1 = VideoReader(filename=filename3)
vr1.setNextFrame(60)
frameskip = 65
vr2 = VideoReader(filename=filename3)
vr2.setNextFrame(frameskip)

ret1, frame1 = vr1.readFrame()
ret2, frame2 = vr2.readFrame()

cv2.imshow("1", frame1)
cv2.imshow("2", frame2)

bf = BallFinder()

filt = bf.rgbDiff(frame1, frame2)
cv2.imshow("filt", filt)

hsvfilt = bf.hsvFilt(frame1)
cv2.imshow("hsv", hsvfilt)

hsvDiff = bf.hsvDiff(frame1, frame2, withFilt=False)
cv2.imshow("hsvdiff", hsvDiff)

test = cv2.bitwise_and(hsvfilt, hsvfilt, mask=filt)
cv2.imshow("t" , test)

foundBall = bf.calcBallCenter(test)
#ballFrame = bf.drawBallOnFrame(frame1)
#cv2.imshow("f" , ballFrame)

test = bf.GetCornernessMask(frame1, frame2)
cv2.imshow("test", test)


frameDiff1 = bf.hsvDiff(frame1, frame2)
frameDiff2 = bf.rgbDiff(frame1, frame2)
frameCornerMask = bf.GetCornernessMask(frame1, frame2)
mask = bf.aveMask(frameDiff1, frameDiff2, frameCornerMask)
cv2.imshow("mask", mask)
