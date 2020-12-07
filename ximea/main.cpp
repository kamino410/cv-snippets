#include <opencv2/opencv.hpp>
#include <xiApi.h>

// works at around 90 FPS (97 FPS if disable imshow)

#define HandleResult(res, place)                 \
  if (res != XI_OK) {                            \
    printf("Error after %s (%d)\n", place, res); \
    exit(1);                                     \
  }

int main() {
  HANDLE xiH = NULL;
  XI_RETURN stat = XI_OK;

  // Allocate memory for ximea image
  XI_IMG img;
  memset(&img, 0, sizeof(img));
  img.size = sizeof(XI_IMG);

  // Connect to camera
  stat = xiOpenDevice(0, &xiH);
  HandleResult(stat, "xiOpenDevice");

  // Configuration
  stat = xiSetParamInt(xiH, XI_PRM_EXPOSURE, 100000);
  HandleResult(stat, "xiGetParam (exposure)");
  stat = xiSetParamFloat(xiH, XI_PRM_GAIN, 1);
  HandleResult(stat, "xiGetParam (gain)");
  stat = xiSetParamFloat(xiH, XI_PRM_GAMMAY, 1.0);
  HandleResult(stat, "xiGetParam (gamma y)");

  // Start acquisition
  stat = xiStartAcquisition(xiH);
  HandleResult(stat, "xiStartAcquisition");

  // Get image from camera to check the image size
  stat = xiGetImage(xiH, 5000, &img);
  HandleResult(stat, "xiGetImage");
  
  // Share the buffer between XI_IMG and cv::Mat (Ximea API uses a same buffer in each frame)
  cv::Mat cvimg(cv::Size((int)img.width, (int)img.height), CV_8UC1);
  cvimg.data = (unsigned char*)img.bp;
  
  // Preview output from camera
  do {
    // Get image from camera
    stat = xiGetImage(xiH, 5000, &img);
    HandleResult(stat, "xiGetImage");

    // Share the buffer between XI_IMG and cv::Mat
    cv::Mat cvimg(cv::Size((int)img.width, (int)img.height), CV_8UC1);
    cvimg.data = (unsigned char*)img.bp;

    // Show
    cv::imshow("test", cvimg);
  } while (cv::waitKey(1) == -1);

  // Terminate
  xiStopAcquisition(xiH);
  xiCloseDevice(xiH);
}
