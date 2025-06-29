import cv2
import numpy as np
import glob
import os

# --- パラメータ設定 ---

# チェッカーボードの設定
CHECKERBOARD_SIZE = (9, 6)  # チェッカーボードの内部の角の数 (横, 縦)
SQUARE_SIZE = 25  # チェッカーボードの1マスのサイズ(mm)。相対的なスケールなので、単位は重要ではない。

# 画像パス
CALIB_IMG_DIR = 'checkerboard/'
LEFT_IMG_PATH = 'data/IMG_8301.jpg'
RIGHT_IMG_PATH = 'data/IMG_8302.jpg'
OUTPUT_FILENAME = 'gimini_disparity_map.jpg'

# ステレオマッチング(SGBM)のパラメータ
# これらの値は画像によって調整が必要です
MIN_DISPARITY = 0
NUM_DISPARITIES = 128  # 16の倍数
BLOCK_SIZE = 5
P1 = 8 * 3 * BLOCK_SIZE**2
P2 = 32 * 3 * BLOCK_SIZE**2
DISP_12_MAX_DIFF = 1
UNIQUENESS_RATIO = 10
SPECKLE_WINDOW_SIZE = 100
SPECKLE_RANGE = 32


def calibrate_camera(calib_img_dir):
    """
    チェッカーボード画像を使用してステレオカメラをキャリブレーションする関数
    """
    print("キャリブレーションを開始します...")
    
    # チェッカーボードの3D座標を準備 (z=0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # 画像から検出された3D点と2D点を格納する配列
    objpoints = []  # 3D点 (ワールド座標系)
    imgpoints_l = []  # 2D点 (左画像)
    imgpoints_r = []  # 2D点 (右画像)

    # キャリブレーション用画像を読み込む
    left_images = sorted(glob.glob(os.path.join(calib_img_dir, 'left_*.png')))
    right_images = sorted(glob.glob(os.path.join(calib_img_dir, 'right_*.png')))

    if not left_images or not right_images or len(left_images) != len(right_images):
        print(f"エラー: '{calib_img_dir}' に適切なキャリブレーション画像ペアが見つかりません。")
        print("'left_xx.jpg' と 'right_xx.jpg' のような命名規則で画像を配置してください。")
        return None

    img_shape = None
    for left_fname, right_fname in zip(left_images, right_images):
        img_l = cv2.imread(left_fname)
        img_r = cv2.imread(right_fname)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        # チェッカーボードの角を見つける
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_SIZE, None)

        # 両方の画像で角が検出された場合
        if ret_l and ret_r:
            print(f"{os.path.basename(left_fname)} と {os.path.basename(right_fname)} で角を検出しました。")
            objpoints.append(objp)
            
            # 角の精度を高める
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners2_l)
            
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners2_r)
        else:
            print(f"{os.path.basename(left_fname)} または {os.path.basename(right_fname)} で角を検出できませんでした。スキップします。")

    if not objpoints:
        print("エラー: 有効なチェッカーボードの角がどの画像からも検出できませんでした。")
        return None

    print(f"{len(objpoints)}ペアの有効な画像からキャリブレーションを実行します...")

    # ステレオキャリブレーション
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, None, None, None, None, img_shape,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    if ret:
        print("キャリブレーション成功！")
        return {
            "mtx_l": mtx_l, "dist_l": dist_l, "mtx_r": mtx_r, "dist_r": dist_r,
            "R": R, "T": T, "img_shape": img_shape
        }
    else:
        print("エラー: キャリブレーションに失敗しました。")
        return None


def compute_disparity_map(left_img_path, right_img_path, calib_data):
    """
    キャリブレーションデータを使用して視差マップを計算する関数
    """
    print("視差マップの計算を開始します...")
    
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    
    if img_l is None or img_r is None:
        print(f"エラー: {left_img_path} または {right_img_path} を読み込めません。")
        return

    # 平行化（Rectification）のための変換を計算
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib_data['mtx_l'], calib_data['dist_l'],
        calib_data['mtx_r'], calib_data['dist_r'],
        calib_data['img_shape'], calib_data['R'], calib_data['T'],
        alpha=0  # alpha=0で歪み補正後に黒い部分がなくなるように、alpha=1で全てのピクセルが残るように
    )

    # 歪み補正と平行化のためのマップを作成
    map_l1, map_l2 = cv2.initUndistortRectifyMap(calib_data['mtx_l'], calib_data['dist_l'], R1, P1, calib_data['img_shape'], cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(calib_data['mtx_r'], calib_data['dist_r'], R2, P2, calib_data['img_shape'], cv2.CV_32FC1)

    # 画像に変換を適用
    img_l_rectified = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    img_r_rectified = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)
    
    # グレースケールに変換
    gray_l_rect = cv2.cvtColor(img_l_rectified, cv2.COLOR_BGR2GRAY)
    gray_r_rect = cv2.cvtColor(img_r_rectified, cv2.COLOR_BGR2GRAY)

    # SGBM (Semi-Global Block Matching) で視差を計算
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1,
        P2=P2,
        disp12MaxDiff=DISP_12_MAX_DIFF,
        uniquenessRatio=UNIQUENESS_RATIO,
        speckleWindowSize=SPECKLE_WINDOW_SIZE,
        speckleRange=SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    print("SGBMアルゴリズムで視差を計算中...")
    disparity = stereo.compute(gray_l_rect, gray_r_rect).astype(np.float32) / 16.0
    
    # 視差マップを正規化して表示・保存
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    print(f"計算完了！視差マップを '{OUTPUT_FILENAME}' に保存します。")
    cv2.imwrite(OUTPUT_FILENAME, disparity_normalized)

    # 結果を表示
    cv2.imshow('Disparity Map', disparity_normalized)
    print("表示ウィンドウで 'q' キーを押すと終了します。")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. カメラキャリブレーション
    calibration_data = calibrate_camera(CALIB_IMG_DIR)

    # 2. 視差マップの計算
    if calibration_data:
        compute_disparity_map(LEFT_IMG_PATH, RIGHT_IMG_PATH, calibration_data)
    else:
        print("キャリブレーションに失敗したため、視差マップの計算を中止しました。")
        print(f"'{CALIB_IMG_DIR}' フォルダの画像を確認してください。")