#include <iostream>
#include <vector>
#include <sstream>

#include <opencv2/opencv.hpp>  // checked at opencv 3.4.1
#include <opencv2/structured_light.hpp>

#define WINDOWWIDTH 1920
#define WINDOWHEIGHT 1080
#define GRAYCODEWIDTHSTEP 5
#define GRAYCODEHEIGHTSTEP 5
#define GRAYCODEWIDTH WINDOWWIDTH / GRAYCODEWIDTHSTEP
#define GRAYCODEHEIGHT WINDOWHEIGHT / GRAYCODEHEIGHTSTEP
#define WHITETHRESHOLD 5
#define BLACKTHRESHOLD 40

// 自分が使うカメラに合わせて実装する
void initializeCamera() {}
cv::Mat getCameraImage() {
  // return img.clone(); のようにclone()しなければいけないケースがあるので注意
  // グレイスケール画像を返すように実装
}
void terminateCamera() {}

// Camera to Projector
struct C2P {
  int cx;
  int cy;
  int px;
  int py;
  C2P(int camera_x, int camera_y, int proj_x, int proj_y) {
    cx = camera_x;
    cy = camera_y;
    px = proj_x;
    py = proj_y;
  }
};

void main() {
  // -----------------------------------
  // ----- Prepare graycode images -----
  // -----------------------------------
  cv::structured_light::GrayCodePattern::Params params;
  params.width = GRAYCODEWIDTH;
  params.height = GRAYCODEHEIGHT;
  auto pattern = cv::structured_light::GrayCodePattern::create(params);

  // 用途:decode時にpositiveとnegativeの画素値の差が
  //      常にwhiteThreshold以上である画素のみdecodeする
  pattern->setWhiteThreshold(WHITETHRESHOLD);
  // 用途:ShadowMask計算時に white - black > blackThreshold
  //      ならば前景（グレイコードを認識した）と判別する
  // 今回はこれを設定しても参照されることはないが一応セットしておく
  pattern->setBlackThreshold(BLACKTHRESHOLD);

  std::vector<cv::Mat> graycodes;
  pattern->generate(graycodes);

  cv::Mat blackCode, whiteCode;
  pattern->getImagesForShadowMasks(blackCode, whiteCode);
  graycodes.push_back(blackCode), graycodes.push_back(whiteCode);

  // -----------------------------
  // ----- Prepare cv window -----
  // -----------------------------
  cv::namedWindow("Pattern", cv::WINDOW_NORMAL);
  cv::resizeWindow("Pattern", GRAYCODEWIDTH, GRAYCODEHEIGHT);
  // 2枚目のディスプレイにフルスクリーン表示
  cv::moveWindow("Pattern", 1920, 0);
  cv::setWindowProperty("Pattern", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

  // ----------------------------------
  // ----- Wait camera adjustment -----
  // ----------------------------------
  initializeCamera();

  cv::imshow("Pattern", graycodes[graycodes.size() - 3]);
  while (true) {
    cv::Mat img = getCameraImage();
    cv::imshow("camera", img);
    if (cv::waitKey(1) != -1) break;
  }

  // --------------------------------
  // ----- Capture the graycode -----
  // --------------------------------
  std::vector<cv::Mat> captured;
  int cnt = 0;
  for (auto gimg : graycodes) {
    cv::imshow("Pattern", gimg);
    // ディスプレイに表示->カメラバッファに反映されるまで待つ
    // 必要な待ち時間は使うカメラに依存
    cv::waitKey(400);

    // グレイスケールで撮影
    cv::Mat img = getCameraImage();
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << cnt++;
    cv::imwrite("cam_" + oss.str() + ".png", img);

    captured.push_back(img);
  }

  terminateCamera();

  // -------------------------------
  // ----- Decode the graycode -----
  // -------------------------------
  // pattern->decode()は視差マップの解析に使う関数なので今回は使わない
  // pattern->getProjPixel()を使って各カメラ画素に写ったプロジェクタ画素の座標を計算
  cv::Mat white = captured.back();
  captured.pop_back();
  cv::Mat black = captured.back();
  captured.pop_back();

  int camHeight = captured[0].rows;
  int camWidth = captured[0].cols;

  cv::Mat c2pX = cv::Mat::zeros(camHeight, camWidth, CV_16U);
  cv::Mat c2pY = cv::Mat::zeros(camHeight, camWidth, CV_16U);
  std::vector<C2P> c2pList;
  for (int y = 0; y < camHeight; y++) {
    for (int x = 0; x < camWidth; x++) {
      cv::Point pixel;
      if (white.at<cv::uint8_t>(y, x) - black.at<cv::uint8_t>(y, x) > BLACKTHRESHOLD &&
          !pattern->getProjPixel(captured, x, y, pixel)) {
        c2pX.at<cv::uint16_t>(y, x) = pixel.x;
        c2pY.at<cv::uint16_t>(y, x) = pixel.y;
        c2pList.push_back(C2P(x, y, pixel.x * GRAYCODEWIDTHSTEP, pixel.y * GRAYCODEHEIGHTSTEP));
      }
    }
  }

  // ---------------------------
  // ----- Save C2P as csv -----
  // ---------------------------
  std::ofstream os("c2p.csv");
  for (auto elem : c2pList) {
    os << elem.cx << ", " << elem.cy << ", " << elem.px << ", " << elem.py << std::endl;
  }
  os.close();

  // ----------------------------
  // ----- Visualize result -----
  // ----------------------------
  cv::Mat viz = cv::Mat::zeros(camHeight, camWidth, CV_8UC3);
  for (int y = 0; y < camHeight; y++) {
    for (int x = 0; x < camWidth; x++) {
      viz.at<cv::Vec3b>(y, x)[0] = (unsigned char)c2pX.at<cv::uint16_t>(y, x);
      viz.at<cv::Vec3b>(y, x)[1] = (unsigned char)c2pY.at<cv::uint16_t>(y, x);
    }
  }
  cv::imshow("result", viz);
  cv::waitKey(0);
}
