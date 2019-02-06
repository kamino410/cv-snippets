#pragma once

#include <opencv2/opencv.hpp>
#include <FlyCapture2.h>  //C:\Program Files\Point Grey Research\FlyCapture2\include

class FlyCameraWrapper {
  FlyCapture2::Camera cam;

  void printFlyInfo();
  void printCamInfo(FlyCapture2::CameraInfo *pCamInfo);

 public:
  cv::Mat img;

  FlyCameraWrapper();
  ~FlyCameraWrapper();

  bool init(int serial);
  bool autoExposure(bool flag, float absValue = 1.5f);
  bool autoSaturation(bool flag, float absValue = 50.0f);
  bool autoShutter(bool flag, float absValue = 7.5f);
  bool autoGain(bool flag, float absValue = 0.0f);
  bool start();
  bool capture();
};
