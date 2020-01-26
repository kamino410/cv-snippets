#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string val;
  std::vector<std::string> result;
  while (std::getline(stream, val, delimiter)) { result.push_back(val); }
  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage : <path_to_lut_file> <path_to_src_img> <height_of_result_img> "
                 "<width_of_result_img> <path_to_save_img>"
              << std::endl;
    return 1;
  }

  // size of target image
  const int HEIGHT = atoi(argv[3]);
  const int WIDTH = atoi(argv[4]);

  std::cout << "Loading csv ..." << std::endl;
  cv::Mat mapping = cv::Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
  std::ifstream ifs(argv[1]);
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> row = split(line, ',');
    for (int i = 0; i < row.size(); i++) {
      int x = atoi(row[0].c_str());
      int y = atoi(row[1].c_str());
      mapping.at<cv::Vec2f>(y, x)[0] = atof(row[2].c_str());
      mapping.at<cv::Vec2f>(y, x)[1] = atof(row[3].c_str());
    }
  }

  std::cout << "Loading source image ..." << std::endl;
  cv::Mat src = cv::imread(argv[2]);
  if (!src.data) {
    std::cout << "Source iamges was not found!" << std::endl;
    return 1;
  }

  std::cout << "Mapping ..." << std::endl;
  cv::Mat res = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      float u = mapping.at<cv::Vec2f>(y, x)[0];
      float v = mapping.at<cv::Vec2f>(y, x)[1];

      int iu = u;
      int iv = v;

      if (iu < 0 || src.cols <= iu + 1 || iv < 0 || src.rows <= iv + 1) continue;

      float du = u - iu;
      float dv = v - iv;

      float tl = (1.0 - du) * (1.0 - dv);
      float tr = (du) * (1.0 - dv);
      float bl = (1.0 - du) * (dv);
      float br = (du) * (dv);

      for (int c = 0; c < 3; c++) {
        float val = tl * src.at<cv::Vec3b>(iv, iu)[c] + tr * src.at<cv::Vec3b>(iv, iu + 1)[c] +
                    bl * src.at<cv::Vec3b>(iv + 1, iu)[c] +
                    br * src.at<cv::Vec3b>(iv + 1, iu + 1)[c];
        res.at<cv::Vec3b>(y, x)[c] = round(val);
      }
    }
  }
  cv::imwrite(argv[5], res);

  return 0;
}
