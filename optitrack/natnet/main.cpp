#include <iostream>
#include <Eigen/Dense>
#include "optitrack_client.hpp"

// 実行時にNatNetLib.dll, NatNetML.dll, NatNetML.xmlが必要

int main() {
  OptiTrackClient client;
  int cnt = 0;
  while (!client.connect()) {
    if (cnt++ > 20) {
      std::cout << "Failed to open NatNet connection" << std::endl;
      exit(1);
    }
    for (int i = 0; i < 100000000; i++) {
      ;
    }
  }
  for (int c = 0; c < 10; c++) {
    auto body = client.getBody(3);

    Eigen::Quaternion<double> q(body.qw, body.qx, body.qy, body.qz);
    std::cout << "rotation(global -> model)" << std::endl;
    auto mat = q.toRotationMatrix();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        std::cout << mat(i, j) << ", ";
      }
    }
    std::cout << std::endl;
    std::cout << "translation(global -> model)" << std::endl;
    auto vec = Eigen::Vector3d(body.x, body.y, body.z).transpose();
    std::cout << vec.x() << ", " << vec.y() << ", " << vec.z() << std::endl;

    for (int i = 0; i < 100000000; i++) {
      ;
    }
  }

  return 0;
}
