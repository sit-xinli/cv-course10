# 必要なモジュールをインポートする
import cv2
import numpy as np
import glob


# Define the dimensions of checkerboard
CHECKERBOARD = (7, 7)


# 指定された
# 精度εに達するか、または
# 指定された回数の反復が終了した時.
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# 3次元点のベクトル
threedpoints = []

# 2次元点のベクトル
twodpoints = []


#  3Dポイント実世界座標
objectp3d = np.zeros((1, CHECKERBOARD[0] 
                      * CHECKERBOARD[1], 
                      3), np.float32)
# 3Dポイントの座標を設定：z座標は0, x座標とy座標はチェッカーボードのグリッドに基づく
# 単位は通常はミリメートルやセンチメートルだが、最後でピクセル単位になるのでここでどうでもいい
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# 指定されたディレクトリに格納されている個々の画像のパスを抽出する
# パスを抽出する。 
images = glob.glob('data/*.png')

for filename in images:
    image = cv2.imread(filename)
    #元画像が4032×3024大きすぎるため、リサイズして処理を軽くする
    image = cv2.resize(image, ((int)(image.shape[1]/4), (int)(image.shape[0]/4)))
    #計算はカラーに関係ないため、カラーからグレー画像へ
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # チェス盤の角を見つける
    # 希望する角の数が見つかった場合 ret = true
    ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH 
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE)

    # 希望する数のコーナーが検出された場合、
    # ピクセル座標を絞り込んで表示する
    # チェッカーボードの画像に表示する。
    if ret == True:
        threedpoints.append(objectp3d)

        # コーナー検出、ピクセル座標の精度を上げるため、サブピクセル精度.
        corners2 = cv2.cornerSubPix(
            grayColor, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

        # コーナーを描いて表示する
        image = cv2.drawChessboardCorners(image, 
                                          CHECKERBOARD, 
                                          corners2, ret)

    cv2.imshow('img', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# ３D座標とそれに対応する検出されたコーナー点をつかい、
# カメラキャリブレーションを実行する。 
# 複数画像のK行列が同じく手、[R |t]だけ違うので、複数画像があれば
# より精度の高いキャリブレーションができる。
# 今回は5枚の画像を使う。
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

# 必要な出力を表示する
print(" カメラ内部パラメタ：")
print(matrix)

print("\n カメラ変形係数:")
print(distortion)

print("\n 外部変換--回転:")
print(r_vecs)

print("\n 外部変換--並進:")
print(t_vecs)

