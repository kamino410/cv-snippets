#pragma once

#include <opencv2/opencv.hpp>

class MonoCalibration {
  int horiCorner;
  int vertCorner;
  cv::Size imageSize;
  std::vector<cv::Point3f> cornersRel;
  std::vector<std::vector<cv::Point3f>> cornersIn3D;
  std::vector<std::vector<cv::Point2f>> cornersInImgs;

  MonoCalibration() {}

 public:
  cv::Mat imgVisualizeCorner;
  cv::Mat intrinsic;
  cv::Mat dist;

  MonoCalibration(int horiCrossCount, int vertCrossCount, float blockSize);

  bool check_corners(cv::Mat img);
  void calibrate();
  void save_params(std::string filename);
};

class StereoCalibration {
  int horiCorner;
  int vertCorner;

  cv::Size imageSize;
  std::vector<cv::Point3f> cornersRel;
  std::vector<std::vector<cv::Point3f>> cornersIn3D;
  std::vector<std::vector<cv::Point2f>> cornersInImgs1;
  std::vector<std::vector<cv::Point2f>> cornersInImgs2;

  StereoCalibration() {}

 public:
  cv::Mat imgVisualizeCorner1;
  cv::Mat imgVisualizeCorner2;
  cv::Mat intrinsic1;
  cv::Mat intrinsic2;
  cv::Mat dist1;
  cv::Mat dist2;
  cv::Mat rot;
  cv::Mat tvec;

  StereoCalibration(int horiCrossCount, int vertCrossCount, float blockSize);

  bool check_corners(cv::Mat img1, cv::Mat img2);
  void calibrate();
  void save_params(std::string filename);
};
