import cv2
import numpy as np
import glob
import os

# --- パラメータ設定 ---

# チェッカーボードの設定
CHECKERBOARD_SIZE = (9, 6)  # チェッカーボードの内部の角の数 (横, 縦)
SQUARE_SIZE = 0.025  # チェッカーボードの1マスのサイズ(メートル単位)。奥行き計算のために実世界のスケールに合わせる。

# 画像パス
CALIB_IMG_DIR = 'checkboard/'
LEFT_IMG_PATH = 'data/IMG_8301.jpg'
RIGHT_IMG_PATH = 'data/IMG_8302.jpg'
OUTPUT_DISPARITY_FILENAME = 'disparity_map.png'
OUTPUT_DEPTH_FILENAME = 'depth_map.png'

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
    チェッカーボード画像を使用してステレオカメラをキャリブレーションする関数 (改良版)
    最初に各カメラを個別にキャリブレーションし、その結果をステレオキャリブレーションに利用する。
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
    left_images = sorted(glob.glob(os.path.join(calib_img_dir, 'left_*.jpg')))
    right_images = sorted(glob.glob(os.path.join(calib_img_dir, 'right_*.jpg')))

    if not left_images or not right_images or len(left_images) != len(right_images):
        print(f"エラー: '{calib_img_dir}' に適切なキャリブレーション画像ペアが見つかりません。")
        print("'left_xx.jpg' と 'right_xx.jpg' のような命名規則で画像を配置してください。")
        return None

    img_shape = None
    print("チェッカーボードの角を検出中...")
    for left_fname, right_fname in zip(left_images, right_images):
        img_l = cv2.imread(left_fname)
        img_r = cv2.imread(right_fname)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        # findChessboardCorners を使って角を見つける
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_SIZE, None)

        if ret_l and ret_r:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners2_l)
            
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners2_r)

    if not objpoints:
        print("エラー: 有効なチェッカーボードの角がどの画像からも検出できませんでした。")
        return None
    print(f"{len(objpoints)}ペアの有効な画像から角を検出しました。")

    # --- 個別のカメラキャリブレーション ---
    print("\n左カメラをキャリブレーション中...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    print("右カメラをキャリブレーション中...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # --- ステレオキャリブレーション ---
    print("\nステレオキャリブレーションを実行中...")
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_shape, criteria=stereocalib_criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )

    if ret:
        print("キャリブレーション成功！")
        # 平行化のための変換を計算
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T, alpha=0
        )
        return {
            "mtx_l": mtx_l, "dist_l": dist_l, "mtx_r": mtx_r, "dist_r": dist_r,
            "R": R, "T": T, "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q, "img_shape": img_shape
        }
    else:
        print("エラー: ステレオキャリブレーションに失敗しました。")
        return None

def compute_depth_map(left_img_path, right_img_path, calib_data):
    """
    キャリブレーションデータを使用して視差マップとデプスマップを計算する関数
    """
    print("\nデプスマップの計算を開始します...")
    
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    
    if img_l is None or img_r is None:
        print(f"エラー: {left_img_path} または {right_img_path} を読み込めません。")
        return

    # 歪み補正と平行化のためのマップを作成
    map_l1, map_l2 = cv2.initUndistortRectifyMap(calib_data['mtx_l'], calib_data['dist_l'], calib_data['R1'], calib_data['P1'], calib_data['img_shape'], cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(calib_data['mtx_r'], calib_data['dist_r'], calib_data['R2'], calib_data['P2'], calib_data['img_shape'], cv2.CV_32FC1)

    # 画像に変換を適用
    img_l_rectified = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    img_r_rectified = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)
    
    gray_l_rect = cv2.cvtColor(img_l_rectified, cv2.COLOR_BGR2GRAY)
    gray_r_rect = cv2.cvtColor(img_r_rectified, cv2.COLOR_BGR2GRAY)

    # SGBMで視差を計算
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY, numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE,
        P1=P1, P2=P2, disp12MaxDiff=DISP_12_MAX_DIFF, uniquenessRatio=UNIQUENESS_RATIO,
        speckleWindowSize=SPECKLE_WINDOW_SIZE, speckleRange=SPECKLE_RANGE, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    print("SGBMアルゴリズムで視差を計算中...")
    disparity = stereo.compute(gray_l_rect, gray_r_rect).astype(np.float32) / 16.0
    
    # 視差マップを保存
    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(OUTPUT_DISPARITY_FILENAME, disparity_visual)
    print(f"視差マップを '{OUTPUT_DISPARITY_FILENAME}' に保存しました。")

    # --- デプスマップの計算 ---
    print("3Dポイントクラウドとデプスマップを計算中...")
    points_3D = cv2.reprojectImageTo3D(disparity, calib_data['Q'])
    
    # Z座標（奥行き）を抽出
    depth_map = points_3D[:, :, 2]

    # 奥行きの値を正規化して可視化・保存
    # 無限大や無効な値をマスク
    mask = (disparity > disparity.min())
    valid_depth = depth_map[mask]
    
    # 0-5メートルの範囲でクリッピングして可視化
    min_val = 0
    max_val = 5 # 5メートル
    depth_visual = np.uint8(np.clip((depth_map - min_val) / (max_val - min_val) * 255, 0, 255))
    depth_visual[~mask] = 0 # 無効な領域は黒
    depth_visual_color = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    depth_visual_color[~mask] = [0, 0, 0]

    cv2.imwrite(OUTPUT_DEPTH_FILENAME, depth_visual_color)
    print(f"デプスマップを '{OUTPUT_DEPTH_FILENAME}' に保存しました。")

    # 結果を表示
    cv2.imshow('Rectified Left Image', img_l_rectified)
    cv2.imshow('Disparity Map', disparity_visual)
    cv2.imshow('Depth Map', depth_visual_color)
    print("\n表示ウィンドウで 'q' キーを押すと終了します。")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. カメラキャリブレーション
    calibration_data = calibrate_camera(CALIB_IMG_DIR)

    # 2. デプスマップの計算
    if calibration_data:
        compute_depth_map(LEFT_IMG_PATH, RIGHT_IMG_PATH, calibration_data)
    else:
        print("\nキャリブレーションに失敗したため、処理を中止しました。")
        print(f"'{CALIB_IMG_DIR}' フォルダの画像とチェッカーボードの設定を確認してください。")