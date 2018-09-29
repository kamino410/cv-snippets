// Levenberg-Marquardt Method (Least squares problem)

// Sample for cosine curve fitting
//   (a, w, p, c) = minarg |y - f(t)|
//     f(x) = a * cos(w*t + p) + c
//   Truth : a = 2, w = 10, p = pi/2, c = 1

#include <iostream>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace Eigen;

template <typename ScalarType, int InputSize = Dynamic, int ValueSize = Dynamic>
struct Functor {
  typedef ScalarType Scalar;
  enum { InputsAtCompileTime = InputSize, ValuesAtCompileTime = ValueSize };
  typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
};

// ヤコビアンの解析解を与えるパターン
struct FunctorA : Functor<double> {
  FunctorA(int inputSize, int valueSize, const double t[], const double y[])
      : _inputs(inputSize), _values(valueSize), t(t), y(y) {}

  int operator()(const VectorXd& x, VectorXd& res) const {
    for (int i = 0; i < _values; i++) {
      res[i] = y[i] - (x[0] * cos(x[1] * t[i] + x[2]) + x[3]);
    }
    return 0;
  }

  int df(const VectorXd& x, MatrixXd& jac) {
    for (int i = 0; i < _values; i++) {
      jac(i, 0) = -cos(x[1] * t[i] + x[2]);
      double tmp = sin(x[1] * t[i] + x[2]);
      jac(i, 1) = t[i] * tmp;
      jac(i, 2) = tmp;
      jac(i, 3) = -1;
    }
    return 0;
  }

  const double *t, *y;
  const int _inputs, _values;
  int inputs() const { return _inputs; }
  int values() const { return _values; }
};

// ヤコビアンを数値計算させるパターン
struct FunctorB : Functor<double> {
  FunctorB(int inputSize, int valueSize, const double t[], const double y[])
      : _inputs(inputSize), _values(valueSize), t(t), y(y) {}

  int operator()(const VectorXd& x, VectorXd& res) const {
    for (int i = 0; i < _values; i++) {
      res[i] = y[i] - (x[0] * cos(x[1] * t[i] + x[2]) + x[3]);
    }
    return 0;
  }

  const double *t, *y;
  const int _inputs, _values;
  int inputs() const { return _inputs; }
  int values() const { return _values; }
};

// データ
const double t[101] = {
    0.,   0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
    0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,  0.21, 0.22, 0.23,
    0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,  0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4,  0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
    0.48, 0.49, 0.5,  0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
    0.6,  0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,  0.71,
    0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,  0.81, 0.82, 0.83,
    0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,  0.91, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99, 1.};
const double y[101] = {
    1.00000000e+00,  8.00333167e-01,  6.02661338e-01,  4.08959587e-01,
    2.21163315e-01,  4.11489228e-02,  -1.29284947e-01, -2.88435374e-01,
    -4.34712182e-01, -5.66653819e-01, -6.82941970e-01, -7.82414720e-01,
    -8.64078172e-01, -9.27116371e-01, -9.70899460e-01, -9.94989973e-01,
    -9.99147206e-01, -9.83329621e-01, -9.47695262e-01, -8.92600175e-01,
    -8.18594854e-01, -7.26418733e-01, -6.16992808e-01, -4.91410424e-01,
    -3.50926361e-01, -1.96944288e-01, -3.10027436e-02, 1.45240240e-01,
    3.30023700e-01,  5.21501342e-01,  7.17759984e-01,  9.16838675e-01,
    1.11674829e+00,  1.31549139e+00,  1.51108220e+00,  1.70156646e+00,
    1.88504089e+00,  2.05967228e+00,  2.22371578e+00,  2.37553232e+00,
    2.51360499e+00,  2.63655422e+00,  2.74315154e+00,  2.83233187e+00,
    2.90320415e+00,  2.95506024e+00,  2.98738201e+00,  2.99984652e+00,
    2.99232922e+00,  2.96490523e+00,  2.91784855e+00,  2.85162936e+00,
    2.76690931e+00,  2.66453488e+00,  2.54552898e+00,  2.41108065e+00,
    2.26253328e+00,  2.10137109e+00,  1.92920436e+00,  1.74775333e+00,
    1.55883100e+00,  1.36432501e+00,  1.16617881e+00,  9.66372199e-01,
    7.66901590e-01,  5.69760024e-01,  3.76917273e-01,  1.90300159e-01,
    1.17732977e-02,  -1.56879529e-01, -3.13973197e-01, -4.57938080e-01,
    -5.87335728e-01, -7.00873241e-01, -7.97416192e-01, -8.75999954e-01,
    -9.35839344e-01, -9.76336468e-01, -9.97086691e-01, -9.97882684e-01,
    -9.78716493e-01, -9.39779622e-01, -8.81461113e-01, -8.04343668e-01,
    -7.09197816e-01, -5.96974225e-01, -4.68794196e-01, -3.25938460e-01,
    -1.69834386e-01, -2.04171292e-03, 1.75763030e-01,  3.61803275e-01,
    5.54220172e-01,  7.51091153e-01,  9.50449149e-01,  1.15030224e+00,
    1.34865356e+00,  1.54352125e+00,  1.73295826e+00,  1.91507179e+00,
    2.08804222e+00};

int main() {
  // ---------------------
  // ----- Functor A -----
  // ---------------------
  std::cout << "Functor A ----------" << std::endl;

  FunctorA functor_a(4, 101, t, y);
  LevenbergMarquardt<FunctorA> lm_a(functor_a);

  // 変数が多いのでそれっぽい値を初期値にしないと大域的最適解に収束しない
  VectorXd xa(4);
  xa << 1.5, 8.0, 1.0, 0.5;

  lm_a.minimize(xa);

  std::cout << "result:" << std::endl << xa << std::endl;

  // ---------------------
  // ----- Functor B -----
  // ---------------------
  std::cout << "Functor B ----------" << std::endl;

  FunctorB functor_b(4, 101, t, y);
  NumericalDiff<FunctorB> numDiff(functor_b);
  LevenbergMarquardt<NumericalDiff<FunctorB> > lm_b(numDiff);

  VectorXd xb(4);
  xb << 1.5, 8.0, 1.0, 0.5;

  lm_b.minimize(xb);

  std::cout << "result:" << std::endl << xa << std::endl;
}