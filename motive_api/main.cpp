// Motive 2.1.0 Finalで動作確認

// 起動時のdll探索場所（Path or このプログラムのexeが置かれるディレクトリ or
// カレントディレクトリ）に以下を追加ないしコピーする必要がある
//   "C:/Program Files/OptiTrack/Motive/lib"のNPTrackingToolsx64.dll
//   "C:/Program Files/OptiTrack/Motive"のdll
//   "C:/Program Files/OptiTrack/Motive"のpluginsフォルダ

// Visual Studioからビルドするときは以下の項目を設定すること
//   プロジェクトのプロパティ > C/C++ >
//   追加のインクルードディレクトリに「"C:/Program Files/OptiTrack/Motive/inc"」
//   プロジェクトのプロパティ > リンカー >
//   追加のライブラリディレクトリに「"C:/Program Files/OptiTrack/Motive/lib"」
//   プロジェクトのプロパティ > リンカー >
//   追加の依存ファイルに「NPTrackingToolsx64.lib」

#include <iostream>
#include "NPTrackingTools.h"

int main() {
  if (TT_Initialize() != NPRESULT_SUCCESS) {
    std::cout << "Failed to initialize Motive API" << std::endl;
    return 1;
  }

  // Motiveのプロファイルを指定する（Motive 1系を使うときは不要）
  if (TT_LoadProfile("C:/ProgramData/OptiTrack/MotiveProfile.motive") !=
      NPRESULT_SUCCESS) {
    std::cout << "Failed to load user profile" << std::endl;
    return 1;
  }

  // 自分のキャリブレーションファイルを指定する（Motive 1系を使うときはTT_LoadProject）
  if (TT_LoadCalibration(
          "D:/Documents/OptiTrack/Session 2018-12-21/Calibration Poor (MeanErr "
          "1.107 mm) 2018-12-21 8.cal") != NPRESULT_SUCCESS) {
    std::cout << "Failed to open calibration file" << std::endl;
    return 1;
  }

  std::cout << "Rigid Bodies" << std::endl;
  for (int i = 0; i < TT_RigidBodyCount(); i++) {
    std::cout << "#" << i << std::endl
              << "  Name : " << TT_RigidBodyName(i) << std::endl;
  }

  for (int k = 0; k < 100; k++) {
    if (TT_Update() == NPRESULT_SUCCESS) {
      for (int i = 0; i < TT_RigidBodyCount(); i++) {
        float x, y, z, qx, qy, qz, qw, yaw, pitch, roll;
        TT_RigidBodyLocation(i, &x, &y, &z, &qx, &qy, &qz, &qw, &yaw, &pitch,
                             &roll);
        std::cout << "#" << i << std::endl;
        std::cout << "  Pos : " << x << ", " << y << ", " << z << std::endl;
        std::cout << "  Quat(x,y,z,w) : " << qx << ", " << qy << ", " << qz
                  << ", " << qw << std::endl;
        std::cout << "  Euler(y,p,r) : " << yaw << ", " << pitch << ", " << roll
                  << std::endl;
      }
    }
    for (int i = 0; i < 100000000; i++) {
      ;
    }
  }

  if (TT_Shutdown() != NPRESULT_SUCCESS) {
    std::cout << "Failed to shutdown" << std::endl;
  }

  return 0;
}
