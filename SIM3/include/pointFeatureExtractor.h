#ifndef CALIBATIONTOOL_POINTFEATUREEXTRACTOR_H
#define CALIBATIONTOOL_POINTFEATUREEXTRACTOR_H
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
#include "voxel_grid_large.h"

typedef pcl::PointXYZI PointType;
using namespace std;
class PointFeatureExtractor {
public:
    PointFeatureExtractor(){};
    ~PointFeatureExtractor(){};
    void preprocess();
    void extract();
    pcl::PointCloud<PointType>::Ptr VoteCriterionExtractor(const vector<int> idxs);
    float VoteCriterion(const pcl::PointCloud<PointType>::Ptr points,
                        const pcl::Normal normal,
                        vector<int> neighbors,
                        int index,
                        float segAngle = 5.0);

    vector<Eigen::Vector3f> ProjectPointsToTangentPlane(
            const pcl::PointCloud<PointType>::Ptr points,
            const pcl::Normal normal,
            const vector<int> neighbors,
            int center_idx);

    void projectPointCloudToSphere(
            pcl::PointCloud<PointType>::Ptr cloud,
            pcl::PointCloud<PointType>::Ptr cloud_projected,
            float centerX, float centerY, float centerZ, float radius);

    vector<bool> visiblePoints(const pcl::PointCloud<PointType>::Ptr points, const Eigen::Matrix3d R, const Eigen::Vector3d t);
    vector<bool> visiblePoints(const Eigen::Matrix3d R, const Eigen::Vector3d t);

    pcl::PointCloud<PointType>::Ptr point_clouds;
    pcl::PointCloud<PointType>::Ptr point_ds;
    pcl::PointCloud<PointType>::Ptr edge_points;
    vector<int> edge_idx;
    string pcdPath;
    double downSamplingScale = 0.01;
    float searchNormal = 0.05;
    float segAngle = 5.0;
    float blankRate = 0.5;
    int searchK = 30;
    float sigma = 0.01;
    float visThr = 0.5;
    Eigen::Matrix3d RInit;
    Eigen::Vector3d tInit;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    double minRange, maxRange;
private:
};


#endif //CALIBATIONTOOL_POINTFEATUREEXTRACTOR_H
