#ifndef CALIBATIONTOOL_OPTIMIZATION_H
#define CALIBATIONTOOL_OPTIMIZATION_H
#include <fstream>
#include <thread>
#include <numeric>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/extract_indices.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <memory>
#include "CeresFactor.h"
#include "pointFeatureExtractor.h"
#include "lineFeatureExtractor.h"

using namespace cv;
typedef pcl::PointXYZI PointType;
class Optimization {
public:
    Optimization(){};
    ~Optimization(){};
    cv::Point2d ProjectPointToImage(const PointType& point, const Eigen::Matrix3d &R, Eigen::Vector3d &t);
    void OptimizationCeres();

    PointFeatureExtractor *PCF;
    LineFeatureExtractor *LF;
    Eigen::Matrix3d RInit, ROut;
    Eigen::Vector3d tInit, tOut;
    Eigen::Matrix3d K;
    float matchingPara;
private:


};


#endif //CALIBATIONTOOL_OPTIMIZATION_H
