#pragma once
#include <Eigen/Dense>

#define VECLEN 10

typedef Eigen::Matrix<double, VECLEN, 1> MyVector;

void add_vector_fixed(const MyVector &in1, const MyVector &in2, MyVector &out);

void add_vector(const Eigen::VectorXd &in1, const Eigen::VectorXd &in2, Eigen::VectorXd &out);
