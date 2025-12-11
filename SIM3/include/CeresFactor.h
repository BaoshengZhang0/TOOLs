#ifndef CALIBATIONTOOL_CERESFACTOR_H
#define CALIBATIONTOOL_CERESFACTOR_H

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>  // 关键：LocalParameterization 定义在此处
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <assert.h>
#include <cmath>
static double residual_error = 0.0;
struct LinePointFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LinePointFactor(Eigen::Vector3d line_,
                    Eigen::Vector3d point_,
                    Eigen::Matrix3d K_,
                    double s_,
                    double s__): line_img(line_), point_lidar(point_), K(K_), s(s_), s_(s__){
    }

    template <typename T> bool operator()(const T *q, const T *t, T *residual) const {
        Eigen::Matrix<T, 3, 1> line_i{T(line_img.x()), T(line_img.y()), T(line_img.z())};
        Eigen::Matrix<T, 3, 1> point_l{T(point_lidar.x()), T(point_lidar.y()), T(point_lidar.z())};
        Eigen::Quaternion<T> q_il{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 1> t_il{t[0], t[1], t[2]};

        Eigen::Matrix<T, 3, 1> ip;
        ip = q_il * point_l + t_il;

        Eigen::Matrix<T, 3, 1> pointProjected = K * ip;
        pointProjected /= pointProjected.z();
        Eigen::Matrix<T, 2, 1> pointPixel;
//        pointPixel(0) = pointProjected.x() + 0.5;
//        pointPixel(1) = pointProjected.y() + 0.5;

        pointPixel(0) = pointProjected.x();
        pointPixel(1) = pointProjected.y();

        residual[0] = abs(line_i(0)*pointPixel(0) + line_i(1)*pointPixel(1) + line_i(2)) / sqrt(line_i(0)*line_i(0) + line_i(1)*line_i(1));
        residual[0] /= s;
        residual[0] *= s_;
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d line_,
                                       const Eigen::Vector3d point_,
                                       const Eigen::Matrix3d K_,
                                       const double s_,
                                       const double s__) {
        return (new ceres::AutoDiffCostFunction<LinePointFactor, 1, 4, 3>(new LinePointFactor(line_, point_, K_, s_, s__)));
    }

    Eigen::Vector3d line_img, point_lidar;
    Eigen::Matrix3d K;
    double s, s_;
};
#endif //CALIBATIONTOOL_CERESFACTOR_H
