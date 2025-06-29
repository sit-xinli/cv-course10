import cv2
import numpy as np
import matplotlib.pyplot as plt


#imgL = cv2.imread("data/sample-left.jpg", cv2.IMREAD_GRAYSCALE)  # left image
#imgR = cv2.imread("data/sample-right.jpg", cv2.IMREAD_GRAYSCALE)  # right image
imgL = cv2.imread("data/tsukuba/tsukuba_3.jpg", cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread("data/tsukuba/tsukuba_5.jpg", cv2.IMREAD_GRAYSCALE)  # right image

# 画像のサイズを取得
h, w = imgL.shape[:2]
# 画像のサイズを揃える
if imgR.shape[:2] != (h, w):
    imgR = cv2.resize(imgR, (w, h))

matchingNUM = 8  # Number of matches to use for fundamental matrix estimation

def get_keypoints_and_descriptors(imgL, imgR):
    """ORB検出器とFLANNマッチャーを使用して、キーポイントとデクリプターを取得する、
       ホモグラフィを計算するのに適した対応するマッチを取得する．
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """Filter matches using the Lowe's ratio test.
    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches


def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:matchingNUM],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img)
    cv2.imwrite("data/ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:matchingNUM]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # しきい値（Threshold）と信頼度（confidence）の値は、妥当な結果が得られるまで、ここで変更することができる
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            ransacReprojThreshold=3,
            confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2


############## 良いキーポイントを見つける ##############
kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)

if not flann_match_pairs or len(flann_match_pairs) < 2:
    print("No matches found. Please check the images or parameters.")
    exit()

good_matches = lowes_ratio_test(flann_match_pairs, 0.2) #最近距離比で良いペアを判定
#draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)


############## 基本行列を計算する ##############
F, _, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)


############## ステレオ整列のホモグラフィ変換を計算 ##############
h1, w1 = imgL.shape
h2, w2 = imgR.shape
thresh = 0
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(points1), np.float32(points2), np.float32(F), imgSize=(w1, h1), threshold=thresh,
)

############## 画像整列 (Rectify) ##############
imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
#cv2.imwrite("undistorted_L.png", imgL_undistorted)
#cv2.imwrite("undistorted_R.png", imgR_undistorted)

############## デプスマップ計算 StereoBM
#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)
#plt.imshow(disparity_BM, "gray")
#plt.colorbar()
#plt.show()

# Set disparity parameters. Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 2
min_disp = -4
max_disp = 12
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=5,
    disp12MaxDiff=2,
    P1=8 * 3 * win_size ** 2,
    P2=32 * 3 * win_size ** 2,
)
disparity_SGBM = stereo.compute(imgL_undistorted, imgR_undistorted)

img_output = np.hstack((imgL,disparity_SGBM))

plt.imshow(img_output, "gray")
plt.colorbar()
plt.show()

cv2.imwrite("data/disparity_SGBM.png", img_output)
