#include "FlyCaptureWrapper.h"

#include <iostream>

#define CHECK(error)                         \
  {                                          \
    if (error != FlyCapture2::PGRERROR_OK) { \
      error.PrintErrorTrace();               \
      return false;                          \
    }                                        \
  }

void FlyCameraWrapper::printFlyInfo() {
  FlyCapture2::FC2Version fc2Version;
  FlyCapture2::Utilities::GetLibraryVersion(&fc2Version);

  std::cout << "FlyCapture2 library version: " << fc2Version.major << "."
            << fc2Version.minor << "." << fc2Version.type << "."
            << fc2Version.build << std::endl;
}

void FlyCameraWrapper::printCamInfo(FlyCapture2::CameraInfo *pCamInfo) {
  std::cout << std::endl;
  std::cout << "*** CAMERA INFORMATION ***" << std::endl;
  std::cout << "Serial number - " << pCamInfo->serialNumber << std::endl;
  std::cout << "Camera model - " << pCamInfo->modelName << std::endl;
  std::cout << "Camera vendor - " << pCamInfo->vendorName << std::endl;
  std::cout << "Sensor - " << pCamInfo->sensorInfo << std::endl;
  std::cout << "Resolution - " << pCamInfo->sensorResolution << std::endl;
  std::cout << "Firmware version - " << pCamInfo->firmwareVersion << std::endl;
  std::cout << "Firmware build time - " << pCamInfo->firmwareBuildTime
            << std::endl
            << std::endl;
}

FlyCameraWrapper::FlyCameraWrapper() {}

FlyCameraWrapper::~FlyCameraWrapper() {
  cam.StopCapture();
  cam.Disconnect();
}

bool FlyCameraWrapper::init(int serial) {
  FlyCapture2::BusManager busMgr;
  unsigned int camNum;
  CHECK(busMgr.GetNumOfCameras(&camNum));

  FlyCapture2::PGRGuid guid;
  busMgr.GetCameraFromSerialNumber(serial, &guid);

  CHECK(cam.Connect(&guid));

  FlyCapture2::CameraInfo camInfo;
  CHECK(cam.GetCameraInfo(&camInfo));
  printCamInfo(&camInfo);

  FlyCapture2::FC2Config config;
  CHECK(cam.GetConfiguration(&config));
  CHECK(cam.SetConfiguration(&config));

  return true;
}

bool FlyCameraWrapper::autoExposure(bool flag, float absValue) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::AUTO_EXPOSURE;
  prop.onOff = true;
  prop.autoManualMode = flag;
  prop.absControl = true;
  prop.absValue = absValue;
  CHECK(cam.SetProperty(&prop));

  return true;
}

bool FlyCameraWrapper::autoSaturation(bool flag, float absValue) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::SATURATION;
  prop.onOff = true;
  prop.autoManualMode = flag;
  prop.absControl = true;
  prop.absValue = absValue;
  CHECK(cam.SetProperty(&prop));

  return true;
}

bool FlyCameraWrapper::autoShutter(bool flag, float absValue) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::SHUTTER;
  prop.autoManualMode = flag;
  prop.absControl = true;
  prop.absValue = absValue;
  CHECK(cam.SetProperty(&prop));

  return true;
}

bool FlyCameraWrapper::autoGain(bool flag, float absValue) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::GAIN;
  prop.autoManualMode = flag;
  prop.absControl = true;
  prop.absValue = absValue;
  CHECK(cam.SetProperty(&prop));

  return true;
}

bool FlyCameraWrapper::start() {
  CHECK(cam.StartCapture());

  return true;
}

bool FlyCameraWrapper::capture() {
  FlyCapture2::Image flyimg;
  CHECK(cam.RetrieveBuffer(&flyimg));

  FlyCapture2::Image flybgr;
  flyimg.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &flybgr);

  cv::Mat(flybgr.GetRows(), flybgr.GetCols(), CV_8UC3, flybgr.GetData())
      .copyTo(img);
  return true;
}
