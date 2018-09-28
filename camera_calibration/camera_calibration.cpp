#include "camera_calibration.hpp"

/// Mono Calibration
/// ------------------
MonoCalibration::MonoCalibration(int horiBlockCount, int vertBlockCount,
                                 float blockSize) {
  horiCorner = horiBlockCount;
  vertCorner = vertBlockCount;

  for (int j = 0; j < horiCorner * vertCorner; j++) {
    cornersRel.push_back(cv::Point3f(blockSize * (j % horiCorner),
                                     blockSize * (j / horiCorner), 0.0f));
  }
}

bool MonoCalibration::check_corners(cv::Mat img) {
  imageSize = cv::Size(img.cols, img.rows);

  std::vector<cv::Point2f> corners;
  bool found =
      cv::findChessboardCorners(img, cv::Size(horiCorner, vertCorner), corners);
  if (found) {
    img.copyTo(imgVisualizeCorner);

    cornersIn3D.push_back(cornersRel);
    cornersInImgs.push_back(corners);

    cv::drawChessboardCorners(imgVisualizeCorner,
                              cv::Size(horiCorner, vertCorner),
                              cv::Mat(corners), true);
  }

  return found;
}

void MonoCalibration::calibrate() {
  std::vector<cv::Mat> rvecs, tvecs;
  intrinsic = cv::Mat(3, 3, CV_64F).clone();
  dist = cv::Mat(8, 1, CV_64F).clone();
  cv::calibrateCamera(cornersIn3D, cornersInImgs, imageSize, intrinsic, dist,
                      rvecs, tvecs);
}

void MonoCalibration::save_params(std::string filename) {
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  fs << "intrinsic" << intrinsic;
  fs << "dist" << dist;
  fs.release();
}

// Stereo Calibration
// --------------------
StereoCalibration::StereoCalibration(int horiBlockCount, int vertBlockCount,
                                     float blockSize) {
  horiCorner = horiBlockCount;
  vertCorner = vertBlockCount;

  for (int j = 0; j < horiCorner * vertCorner; j++) {
    cornersRel.push_back(cv::Point3f(blockSize * (j % horiCorner),
                                     blockSize * (j / horiCorner), 0.0f));
  }
}

bool StereoCalibration::check_corners(cv::Mat img1, cv::Mat img2) {
  imageSize = img1.size();

  std::vector<cv::Point2f> corners1;
  std::vector<cv::Point2f> corners2;
  bool found = cv::findChessboardCorners(img1, cv::Size(horiCorner, vertCorner),
                                         corners1) &&
               cv::findChessboardCorners(img2, cv::Size(horiCorner, vertCorner),
                                         corners2);
  if (found) {
    img1.copyTo(imgVisualizeCorner1);
    img2.copyTo(imgVisualizeCorner2);

    cornersIn3D.push_back(cornersRel);
    cornersInImgs1.push_back(corners1);
    cornersInImgs2.push_back(corners2);

    cv::drawChessboardCorners(imgVisualizeCorner1,
                              cv::Size(horiCorner, vertCorner),
                              cv::Mat(corners1), true);
    cv::drawChessboardCorners(imgVisualizeCorner2,
                              cv::Size(horiCorner, vertCorner),
                              cv::Mat(corners2), true);
  }

  return found;
}

void StereoCalibration::calibrate() {
  std::vector<cv::Mat> rvecs1, tvecs1;
  intrinsic1 = cv::Mat(3, 3, CV_64F).clone();
  dist1 = cv::Mat(5, 1, CV_64F).clone();
  cv::calibrateCamera(cornersIn3D, cornersInImgs1, imageSize, intrinsic1, dist1,
                      rvecs1, tvecs1);

  std::vector<cv::Mat> rvecs2, tvecs2;
  intrinsic2 = cv::Mat(3, 3, CV_64F).clone();
  dist2 = cv::Mat(5, 1, CV_64F).clone();
  cv::calibrateCamera(cornersIn3D, cornersInImgs2, imageSize, intrinsic2, dist2,
                      rvecs2, tvecs2);

  cv::Mat essential, fundamental;
  cv::stereoCalibrate(cornersIn3D, cornersInImgs1, cornersInImgs2, intrinsic1,
                      dist1, intrinsic2, dist2, imageSize, rot, tvec, essential,
                      fundamental);
}

void StereoCalibration::save_params(std::string filename) {
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  fs << "intrinsic1" << intrinsic1;
  fs << "dist1" << dist1;
  fs << "intrinsic2" << intrinsic2;
  fs << "dist2" << dist2;
  fs << "rot" << rot;
  fs << "tvec" << tvec;
  fs.release();
}
