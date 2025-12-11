//
// Created by zbs on 8/9/25.
//

#ifndef CALIBATIONTOOL_CALIBRATION_H
#define CALIBATIONTOOL_CALIBRATION_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include "pointFeatureExtractor.h"
#include "lineFeatureExtractor.h"
#include "optimization.h"

using namespace std;
class Calibration {
public:
    Calibration(){

    }
    ~Calibration(){};
    void ParamerInit(const YAML::Node &config);
    void SetPCFDetector(PointFeatureExtractor *PCF_);
    void SetLFDetector(LineFeatureExtractor *LF_);
    void SetOptimizer(Optimization *OP_);

    Eigen::Vector2d ProjectPointToImage(const Eigen::Vector3d& point);
    void MapCustomColor(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
    void ResultOut();
    string ImgColorPath, ImgDepthPath;
    string PcdPath;
    string OutPath;
    float downSamplingScale;
    float searchNormal;
    float segAngle;
    float blankRate;
    int searchK;
    float sigma;
    float visThr;

    double fx;
    double fy;
    double cx;
    double cy;
//    Eigen::Matrix3d K;
    bool disEnable;
    double k1;
    double k2;
    double k3;
    double p1;
    double p2;
    Eigen::Matrix3d RInit;
    Eigen::Vector3d tInit;

    int lengthThr = 30;
    float distanceThr = 1.5;
    double cannyThr1 = 50;
    double cannyThr2 = 150;
    float depthGradientThr = 30.0;
    PointFeatureExtractor *PCF;
    LineFeatureExtractor *LF;
    Optimization *OP;
    float matchingPara;
private:
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix3d K;
};

#endif //CALIBATIONTOOL_CALIBRATION_H
