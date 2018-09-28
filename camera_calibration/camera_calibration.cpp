#include "camera_calibration.hpp"

/// Mono Calibration
/// ------------------
MonoCalibration::MonoCalibration(int hori_cnt, int vert_corner_cnt,
                                 float block_size) {
  hori = hori_cnt;
  vert = vert_corner_cnt;

  for (int j = 0; j < hori * vert; j++) {
    corners_local.push_back(
        cv::Point3f(block_size * (j % hori), block_size * (j / hori), 0.0f));
  }
}

bool MonoCalibration::check_corners(cv::Mat img) {
  image_size = cv::Size(img.cols, img.rows);

  std::vector<cv::Point2f> corners;
  bool found = cv::findChessboardCorners(img, cv::Size(hori, vert), corners);
  if (found) {
    img.copyTo(img_viz);

    corners_3d.push_back(corners_local);
    corners_imgs.push_back(corners);

    cv::drawChessboardCorners(img_viz, cv::Size(hori, vert), cv::Mat(corners),
                              true);
  }

  return found;
}

void MonoCalibration::calibrate() {
  std::vector<cv::Mat> rvecs, tvecs;
  intr = cv::Mat(3, 3, CV_64F).clone();
  dist = cv::Mat(8, 1, CV_64F).clone();
  cv::calibrateCamera(corners_3d, corners_imgs, image_size, intr, dist, rvecs,
                      tvecs);
}

void MonoCalibration::save_params(std::string filename) {
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  fs << "intr" << intr;
  fs << "dist" << dist;
  fs.release();
}

// Stereo Calibration
// --------------------
StereoCalibration::StereoCalibration(int hori_cnt, int vert_corner_cnt,
                                     float block_size) {
  hori = hori_cnt;
  vert = vert_corner_cnt;

  for (int j = 0; j < hori * vert; j++) {
    corners_local.push_back(
        cv::Point3f(block_size * (j % hori), block_size * (j / hori), 0.0f));
  }
}

bool StereoCalibration::check_corners(cv::Mat img1, cv::Mat img2) {
  image_size = img1.size();

  std::vector<cv::Point2f> corners1;
  std::vector<cv::Point2f> corners2;
  bool found =
      cv::findChessboardCorners(img1, cv::Size(hori, vert), corners1) &&
      cv::findChessboardCorners(img2, cv::Size(hori, vert), corners2);
  if (found) {
    img1.copyTo(img_viz1);
    img2.copyTo(img_viz2);

    corners_3d.push_back(corners_local);
    corners_imgs1.push_back(corners1);
    corners_imgs2.push_back(corners2);

    cv::drawChessboardCorners(img_viz1, cv::Size(hori, vert), cv::Mat(corners1),
                              true);
    cv::drawChessboardCorners(img_viz2, cv::Size(hori, vert), cv::Mat(corners2),
                              true);
  }

  return found;
}

void StereoCalibration::calibrate() {
  std::vector<cv::Mat> rvecs1, tvecs1;
  intr1 = cv::Mat(3, 3, CV_64F).clone();
  dist1 = cv::Mat(5, 1, CV_64F).clone();
  cv::calibrateCamera(corners_3d, corners_imgs1, image_size, intr1, dist1,
                      rvecs1, tvecs1);

  std::vector<cv::Mat> rvecs2, tvecs2;
  intr2 = cv::Mat(3, 3, CV_64F).clone();
  dist2 = cv::Mat(5, 1, CV_64F).clone();
  cv::calibrateCamera(corners_3d, corners_imgs2, image_size, intr2, dist2,
                      rvecs2, tvecs2);

  cv::Mat essential, fundamental;
  cv::stereoCalibrate(corners_3d, corners_imgs1, corners_imgs2, intr1, dist1,
                      intr2, dist2, image_size, rot, tvec, essential,
                      fundamental);
}

void StereoCalibration::save_params(std::string filename) {
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  fs << "intr1" << intr1;
  fs << "dist1" << dist1;
  fs << "intr2" << intr2;
  fs << "dist2" << dist2;
  fs << "rot" << rot;
  fs << "tvec" << tvec;
  fs.release();
}
