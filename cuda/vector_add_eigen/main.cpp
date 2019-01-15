#include <iostream>
#include <Eigen/Dense>
#include "kernel.hpp"

int main() {
  // Fixed size vector
  {
    MyVector a, b;
    a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    b << 1, 3, 5, 7, 9, 11, 13, 15, 17, 19;
    std::cout << (a + b).transpose() << std::endl;  // CPU

    MyVector c;
    add_vector_fixed(a, b, c);
    std::cout << c.transpose() << std::endl;  // GPU
  }

  // Free size vector
  {
    Eigen::VectorXd a(10), b(10);
    a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    b << 1, 3, 5, 7, 9, 11, 13, 15, 17, 19;
    std::cout << (a + b).transpose() << std::endl;  // CPU

    Eigen::VectorXd c(10);
    add_vector(a, b, c);
    std::cout << c.transpose() << std::endl;  // GPU
  }
  return 0;
}
