#include <iostream>
#include <vector>

#include <EDSDK.h>             // Checked at EDSDK 3.8
#include <opencv2/opencv.hpp>  // Checked at OpenCV 4.0.0-pre

static bool flag = false;
static EdsBaseRef eventRef = NULL;

static EdsError EDSCALLBACK handler(EdsUInt32 inEvent, EdsBaseRef inRef,
                                    EdsVoid *inContext) {
  flag = true;
  if (inEvent == kEdsObjectEvent_DirItemRequestTransfer) {
    eventRef = inRef;
  } else {
    EdsRelease(inRef);
    eventRef = NULL;
  }
  return EDS_ERR_OK;
}

int execute() {
  EdsError err;

  // -----------------------------------
  // ----- Get reference to camera -----
  // -----------------------------------
  EdsCameraListRef cameraList = NULL;
  err = EdsGetCameraList(&cameraList);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to obtain camera list" << std::endl;
    return false;
  }

  EdsUInt32 count = 0;
  err = EdsGetChildCount(cameraList, &count);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to obtain number of cameras" << std::endl;
    EdsTerminateSDK();
    return false;
  }
  if (count == 0) {
    std::cout << "Camera was not found" << std::endl;
    EdsTerminateSDK();
    return false;
  }

  EdsCameraRef camera = NULL;
  err = EdsGetChildAtIndex(cameraList, 0, &camera);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to get first camera" << std::endl;
    EdsTerminateSDK();
    return false;
  }

  EdsDeviceInfo deviceInfo;
  err = EdsGetDeviceInfo(camera, &deviceInfo);
  if (err != EDS_ERR_OK || camera == NULL) {
    std::cout << "Failed to get device info" << std::endl;
    EdsTerminateSDK();
    return false;
  }
  bool isLegacy = deviceInfo.deviceSubType == 0;

  EdsRelease(cameraList);

  // ------------------------
  // ----- Open session -----
  // ------------------------
  err = EdsOpenSession(camera);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to open session" << std::endl;
    return false;
  }

  // -------------------------
  // ----- Configuration -----
  // -------------------------
  EdsUInt32 quality = EdsImageQuality_LJN;  // Save as normal jpeg
  err = EdsSetPropertyData(camera, kEdsPropID_ImageQuality, 0, sizeof(quality),
                           &quality);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to configure" << std::endl;
    return false;
  }

  EdsUInt32 saveTo = kEdsSaveTo_Host;
  err =
      EdsSetPropertyData(camera, kEdsPropID_SaveTo, 0, sizeof(saveTo), &saveTo);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to configure" << std::endl;
    return false;
  }

  EdsCapacity capacity = {0x7FFFFFFF, 0x1000, 1};
  err = EdsSendStatusCommand(camera, kEdsCameraStatusCommand_UILock, 0);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to configure" << std::endl;
    return false;
  }
  err = EdsSetCapacity(camera, capacity);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to configure" << std::endl;
    return false;
  }
  err = EdsSendStatusCommand(camera, kEdsCameraStatusCommand_UIUnLock, 0);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to configure" << std::endl;
    return false;
  }

  // ------------------------------------
  // ----- Set Object Event Handler -----
  // ------------------------------------
  err = EdsSetObjectEventHandler(camera, kEdsObjectEvent_All, handler, NULL);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to set event handler" << std::endl;
    return false;
  }

  // ---------------------------
  // ----- Camera settings -----
  // ---------------------------
  // Read "EDSDK API Programming Reference" to know property values
  // (ex. ISO 400 -> 0x58)
  EdsUInt32 iso = 0x58;  // ISO 400
  err = EdsSetPropertyData(camera, kEdsPropID_ISOSpeed, 0, sizeof(iso), &iso);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to set ISO" << std::endl;
    return false;
  }

  EdsUInt32 av = 0x4D;  // Av 20
  err = EdsSetPropertyData(camera, kEdsPropID_Av, 0, sizeof(av), &av);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to set Av" << std::endl;
    return false;
  }

  EdsUInt32 tv = 0x48;  // Tv 1/4
  err = EdsSetPropertyData(camera, kEdsPropID_Tv, 0, sizeof(tv), &tv);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to set Tv" << std::endl;
    return false;
  }

  // --------------------------
  // ----- Take a picture -----
  // --------------------------
  err = EdsSendCommand(camera, kEdsCameraCommand_TakePicture, 0);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to take a picture" << std::endl;
    return false;
  }
  std::cout << "set shutter" << std::endl;

  while (!flag) EdsGetEvent();  // wait until finish
  flag = false;
  if (eventRef == NULL) {
    std::cout << "Failed to handle event" << std::endl;
    return false;
  }

  // -------------------------------------------
  // ----- Send the image to host computer -----
  // -------------------------------------------
  EdsDirectoryItemInfo dirItemInfo;
  err = EdsGetDirectoryItemInfo(eventRef, &dirItemInfo);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to obtain directory" << std::endl;
    return false;
  }

  EdsStreamRef stream = NULL;

  // Case 1 : Save image as file
  // -----------------------------
  // err = EdsCreateFileStream(dirItemInfo.szFileName,
  // kEdsFileCreateDisposition_CreateAlways, kEdsAccess_ReadWrite, &stream); if
  // (err != EDS_ERR_OK) { 	std::cout << "Failed to create file stream" <<
  // std::endl; 	return false;
  //}

  // err = EdsCreateMemoryStream(dirItemInfo.size, &stream);
  // if (err != EDS_ERR_OK) {
  //	std::cout << "Failed to create file stream" << std::endl;
  //	return false;
  //}

  // err = EdsDownload(eventRef, dirItemInfo.size, stream);
  // if (err != EDS_ERR_OK) {
  //	std::cout << "Failed to download" << std::endl;
  //	return false;
  //}
  // -----------------------------

  // Case 2 : Convert the image into cv::Mat (only when save as jpeg)
  // ------------------------------------------------------------------
  err = EdsCreateMemoryStream(dirItemInfo.size, &stream);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to create file stream" << std::endl;
    return false;
  }

  err = EdsDownload(eventRef, dirItemInfo.size, stream);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to download" << std::endl;
    return false;
  }

  err = EdsDownloadComplete(eventRef);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to download" << std::endl;
    return false;
  }

  unsigned char *data = NULL;
  err = EdsGetPointer(stream, (EdsVoid **)&data);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to get pointer" << std::endl;
    return false;
  }

  EdsUInt64 size = 0;
  err = EdsGetLength(stream, &size);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to get image size" << std::endl;
    return false;
  }

  std::vector<unsigned char> buffer(data, data + size);
  cv::Mat img = cv::imdecode(buffer, cv::ImreadModes::IMREAD_COLOR);
  // `img` should be used after release of `eventRef` and `stream`
  // -----------------------------------------

  err = EdsRelease(eventRef);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to release" << std::endl;
    return false;
  }
  eventRef = NULL;

  err = EdsRelease(stream);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to release" << std::endl;
    return false;
  }

  // ---------------------
  // ----- Terminate -----
  // ---------------------
  err = EdsRelease(camera);
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to release" << std::endl;
    return false;
  }
  std::cout << "released" << std::endl;

  cv::imshow("test", img);
  cv::waitKey();

  return true;
}

int main() {
  EdsError err;

  err = EdsInitializeSDK();
  if (err != EDS_ERR_OK) {
    std::cout << "Failed to load SDK" << std::endl;
    return 1;
  }

  if (!execute()) {
    std::cout << "An error ocurred" << std::endl;
  }

  EdsTerminateSDK();

  // getchar();
  return 0;
}
