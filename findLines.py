
import numpy as np
from SourceCode.VideoReader import VideoReader
import cv2
import matplotlib.pyplot as plt
import copy as cp
import scipy.ndimage as ndi
from sklearn.cluster import KMeans

'''

My own attempt at finding the lines.

Select lines using white mask, then can check if the lines are valid by being adjacent to a court colour
within a threshold. Or use the court mask from Stanford to select a region of interest and just find those lines.


'''


def RhoThetaIsect(rho1, rho2, theta1, theta2):
    term1 = rho2 / np.sin(theta2);
    term2 = rho1 / np.sin(theta1);
    term3 = 1.0 / np.tan(theta2) - 1.0 / np.tan(theta1);
    x = (term1 - term2) / term3;
    y = (rho1 - x * np.cos(theta1)) / np.sin(theta1);
    return (int(x), int(y));


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]




filename = "StereoClips/stereoClip1_Kyle.mov"

filename2 = "HomeClips/Dad/behindFence/middleRally2_Dad.mp4"
filename3 = "HomeClips/edited/Dad/rally1-edit_Dad.mp4"

vr = VideoReader(filename=filename2)
vr.setNextFrame(65)
ret, frame = vr.readFrame()
##cv2.imshow("frame", frame)
num_frames = vr.numFrames
height, width = frame.shape[:2]


'''
# Take a small window from the center of the image and average its pixels in HSV
cent_x = int(width / 2);
cent_y = int(height / 2);
cent_win_sz = int(width / 20);
win =  frame[(cent_y - cent_win_sz):(cent_y + cent_win_sz), (cent_x - cent_win_sz):(cent_x + cent_win_sz)];
'''
#30,25,75
smooth = cv2.bilateralFilter(frame,30,25,75)
#cv2.imshow( "smooth", smooth)
lower = np.uint8([230, 200, 200])
upper = np.uint8([255, 255, 255])
test = cv2.inRange(smooth, lower, upper)
#cv2.imwrite("UntrackedFiles/best_line_mask.png", test)
#cv2.imshow( "test", test)

test_frame = cv2.bitwise_and(frame, frame, mask=test)

dst = cv2.cornerHarris(test,2,3,0.04)
corners = cp.copy(frame)
corners[dst>0.01*dst.max()]=[0,0,255]
#cv2.imshow("corners", corners)



court_mask = cv2.morphologyEx(test, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(20,10)))
#cv2.imshow("court_mask", court_mask)

final_mask = cv2.bitwise_and(test, court_mask)
#cv2.imwrite("saved_plots/line-mask_post_MORPHCLOSE.png", and_result)
#cv2.imshow("and", final_mask)

''' Get contours
cont, hier = cv2.findContours(final_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
c = max(cont, key = cv2.contourArea)
sorted_cs = sorted(cont, key=cv2.contourArea)
x = cv2.drawContours(frame, sorted_cs, -1, 255, 2)
x = cp.copy(frame)
for contour in cont:
    if cv2.contourArea(contour) < -1:
        cv2.drawContours(x, contour, -1, (255, 255, 255), 3)

cv2.imshow("x", x)
'''

final_mask = test
minLineLength = 2
maxLineGap = 10
lines = cv2.HoughLines(final_mask, 1,np.pi/360, int(width/8))

for line in lines:
    for rho, theta in line:
        if rho < 0:
            rho *= -1
            theta -= np.pi
print(len(lines))

def getHoughAngleIntersect(theta1, theta2):
    m1 = -np.cos(theta1)/np.sin(theta1)
    m2 = -np.cos(theta2)/np.sin(theta2)
    tanTheta = (m2 - m1) / (1 + m1*m2)
    phi = np.arctan(tanTheta)
    return phi


#clustering :
x_data = lines.reshape(lines.shape[0], lines.shape[2])
kmeans = KMeans(n_clusters=9, random_state=0).fit(x_data)

rhos = [lin[0][0] for lin in lines]
thetas = [lin[0][1] for lin in lines]
centers = kmeans.cluster_centers_
cent_rho = [cen[0] for cen in centers]
cent_th = [cen[1] for cen in centers]
plt.scatter(rhos, thetas, color='b', marker='+')
print(f"Angle intersection: {getHoughAngleIntersect(thetas[0], thetas[3])}  , diff = {thetas[0]-thetas[3]}, angs= {thetas[0]}, {thetas[3]}")
plt.scatter(cent_rho, cent_th, color='r')

xs = [rho * np.cos(theta) for rho, theta in zip(rhos,thetas)]
ys = [rho * np.sin(theta) for rho, theta in zip(rhos,thetas)]
data = [[ rho * np.cos(theta), rho * np.sin(theta)] for rho, theta in zip(rhos, thetas) ]

kmeans = KMeans(n_clusters=9, random_state=0).fit(data)
centers = kmeans.cluster_centers_
cent_x = [cen[0] for cen in centers]
cent_y = [cen[1] for cen in centers]
#plt.scatter(xs, ys, marker='+')
#plt.scatter(cent_x, cent_y, color='r')



#try two passes, one with big threshold and one with small


#Lines lie on horizontal and vertical sections
#Split lines that are almost horizontal
#loop over and extract the horizontal lines in order of confidence

horizontal = []
vertical = []
threshold = 0.15
horiz_indic = []
vert_indic = []
i= 0
for line in lines:
    for rho, theta in line:
        if(theta > (np.pi/2 - threshold)) and (theta < (np.pi/2 +threshold)):
            horizontal.append(line)
            horiz_indic.append(i)
        else:
            vertical.append(line)
            vert_indic.append(i)
        i+=1

#Use to set different thresholds in vert/horizontal axis
strong_horiz = np.zeros([4,1,2])
strong_vert = np.zeros([5,1,2])
strong_lines = np.zeros([9,1,2])
filled = 0
horiz_fill = 0
vert_fill = 0
baseline = False
i=0
for line in lines:

    if i in horiz_indic:
        for rho, theta in line:
            if horiz_fill == 0:
                strong_lines[filled] = line
                strong_horiz[horiz_fill]= line
                if rho > 950:
                    baseline = True
                horiz_fill += 1
                filled += 1
            else:
                closeness_rho = np.isclose(rho, strong_horiz[0:horiz_fill, 0, 0], atol=50)
                closeness_theta = np.isclose(theta, strong_lines[0:horiz_fill, 0, 1], atol=0.15)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)

                if not any(closeness) and horiz_fill < len(strong_horiz):
                    if rho > 950 and baseline == True:
                        continue
                    if rho > 950 and baseline == False:
                        baseline = True
                    strong_lines[filled] = line
                    strong_horiz[horiz_fill] = line
                    horiz_fill += 1
                    filled += 1

    if i in vert_indic:
        for rho, theta in line:
            if vert_fill == 0:
                strong_lines[filled] = line
                strong_vert[vert_fill] = line

                vert_fill += 1
                filled += 1
            else:
                closeness_rho = np.isclose(rho, strong_vert[0:vert_fill, 0, 0], atol=50)
                closeness_theta = np.isclose(theta, strong_vert[0:vert_fill, 0, 1], atol=0.5)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and vert_fill < len(strong_vert):
                    strong_lines[filled] = line
                    strong_vert[vert_fill] = line
                    vert_fill += 1
                    filled += 1
    i+= 1




col = (255,0,0)
rhos_str = []
thetas_str = []
draw_lines = cp.copy(frame)
for line in strong_lines:

    line = np.squeeze(line);
    rho = line[0];
    theta = line[1] + 0.0001;  # hack, ensures eventual intersection...
    rhos_str.append(rho)
    thetas_str.append(theta)
    isect1 = RhoThetaIsect(rho, 0, theta, 0.001);  # vertical
    isect2 = RhoThetaIsect(rho, 0, theta, np.pi / 2);  # horiz
    isect3 = RhoThetaIsect(rho, height, theta, np.pi / 2);  # horiz
    cv2.line(draw_lines, isect1, isect2, col, 2);
    if (isect2[0] > isect3[0]):
        cv2.line(draw_lines, isect1, isect2, col, 2);
    else:
        cv2.line(draw_lines, isect1, isect3, col, 2);

#cv2.imwrite("UntrackedFiles/test_stronglines.png", draw_lines)
cv2.imshow("lines", draw_lines)
#plt.scatter(rhos_str, thetas_str, color='b', marker='+')



isects = []

for line1 in strong_vert:
    for line2 in strong_horiz:
        inter = intersection(line1,line2)
        isects.append(inter)
isects = np.array(isects)
for z in isects:
    cv2.circle(draw_lines, (z[0], z[1]), 5, (0,0,255), 3)
cv2.imwrite("saved_plots/intersections_found.png", draw_lines)
cv2.imshow("points", draw_lines)



#-----------------------------------------------------------------------------------------------------------------------








gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_sm = cv2.bilateralFilter(gray,30,25,75)
#cv2.imshow("gray_sm", gray_sm)
lower = np.uint8([225])
upper = np.uint8([255])
gray_mask = cv2.inRange(gray_sm, lower, upper)
#cv2.imshow("gray_mask", gray_mask)





hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
#cv2.imshow("hsv", hsv)


hls = cv2.cvtColor(smooth, cv2.COLOR_BGR2HLS)
#cv2.imshow("hls", hls)
lower = np.uint8([  0, 220,   0])
upper = np.uint8([255, 255, 255])
hls_mask = cv2.inRange(hls, lower, upper)
#cv2.imshow("hls_mask", hls_mask)

hls_open = cv2.morphologyEx(hls_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,2)))
#cv2.imshow("hls_open", hls_open)


