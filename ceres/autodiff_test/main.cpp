#include <ceres/ceres.h>
#include <iostream>

struct MyFunc {
  template <typename T>
  bool operator()(const T* const param, T* res) const {
    res[0] = sin(param[0]) + param[0] * param[1];
    res[1] = sin(param[0] * param[1]) / param[0];
    return true;
  }
};

int main() {
  // f1 = sin(x) + xy
  // f2 = sin(xy) / x
  ceres::CostFunction* func = new ceres::AutoDiffCostFunction<MyFunc, 2, 2>(new MyFunc());

  const double params[] = {3.1415926536, 2};
  double const* const p = params;

  double res[] = {0, 0};

  double jac_1[] = {0, 0};
  double jac_2[] = {0, 0};
  double* jac[] = {jac_1, jac_2};

  func->Evaluate(&p, res, jac);

  // f1 (pi, 2) = 1 + 2pi
  // f2 (pi, 2) = 0
  std::cout << "f1 (pi, 2) = " << res[0] << std::endl;
  std::cout << "f2 (pi, 2) = " << res[1] << std::endl << std::endl;

  // df1/dx (pi, 2) = 1
  // df1/dy (pi, 2) = pi
  std::cout << "df1/dx (pi, 2) = " << jac[0][0] << std::endl;
  std::cout << "df1/dy (pi, 2) = " << jac[0][1] << std::endl;

  // df2/dx (pi, 2) = 2/pi
  // df2/dy (pi, 2) = 1
  std::cout << "df2/dx (pi, 2) = " << jac[1][0] << std::endl;
  std::cout << "df2/dy (pi, 2) = " << jac[1][1] << std::endl;

  return 0;
}
