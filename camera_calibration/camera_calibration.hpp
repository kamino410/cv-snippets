#pragma once

#include <opencv2/opencv.hpp>

class MonoCalibration {
  int hori, vert;
  cv::Size image_size;
  std::vector<cv::Point3f> corners_local;
  std::vector<std::vector<cv::Point3f>> corners_3d;
  std::vector<std::vector<cv::Point2f>> corners_imgs;

  MonoCalibration() {}

 public:
  cv::Mat img_viz;
  cv::Mat intr;
  cv::Mat dist;

  MonoCalibration(int, int, float);

  bool check_corners(cv::Mat img);
  void calibrate();
  void save_params(std::string filename);
};

class StereoCalibration {
  int hori, vert;
  cv::Size image_size;
  std::vector<cv::Point3f> corners_local;
  std::vector<std::vector<cv::Point3f>> corners_3d;
  std::vector<std::vector<cv::Point2f>> corners_imgs1;
  std::vector<std::vector<cv::Point2f>> corners_imgs2;

  StereoCalibration() {}

 public:
  cv::Mat img_viz1;
  cv::Mat img_viz2;
  cv::Mat intr1;
  cv::Mat intr2;
  cv::Mat dist1;
  cv::Mat dist2;
  cv::Mat rot;
  cv::Mat tvec;

  StereoCalibration(int, int, float);

  bool check_corners(cv::Mat img1, cv::Mat img2);
  void calibrate();
  void save_params(std::string filename);
};
