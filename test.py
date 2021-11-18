import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('RGB2.JPG')
imgRGB = cv2.resize(img1, (600, 450))
img1 = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)


img2 = cv2.imread('NIR2.JPG')
imgNIR = cv2.resize(img2, (600, 450))
img2 = imgNIR[:, :, 2]

akaze = cv2.xfeatures2d.SIFT_create()

kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)
print(f"Keypoints RGB: {len(kp1)}, {des1.shape}")
print(f"Keypoints NIR: {len(kp2)}, {des2.shape}")

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)
print(f"Matches: {len(matches)}")
good_matches = []

for m,n in matches:

    if m.distance < 0.6*n.distance:
        good_matches.append([m])

print(f"Good matches: {len(good_matches)}")
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
print(len(ref_matched_kpts), len(sensed_matched_kpts))
# Compute homography
H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,5.0)

# Warp image
warped_image = cv2.warpPerspective(imgNIR, H, (img2.shape[1], img2.shape[0]))
imgRGB[:, :, 2] = imgNIR[:, :, 0]
imo = imgRGB.copy()

imgRGB[:, :, 2] = warped_image[:, :, 0]
t = np.hstack((imo, imgRGB))
cv2.imshow('result', t)
cv2.waitKey(0)
cv2.destroyAllWindows()
