#include "pointFeatureExtractor.h"
void PointFeatureExtractor::preprocess(){
    point_clouds = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>(pcdPath, *point_clouds) == -1){
        std::cerr << "Couldn't read PCD file." << std::endl;
        return;
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*point_clouds, *point_clouds, indices);
    point_ds = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    if(downSamplingScale > 0.0){
        pcl::VoxelGridLarge<PointType> voxelGrid;
        voxelGrid.setInputCloud(point_clouds);
        voxelGrid.setLeafSize(downSamplingScale, downSamplingScale, downSamplingScale);
        voxelGrid.filter(*point_ds);
    }else{
        *point_ds = *point_clouds;
    }
}
/**
 * Feature Extraction
 */
void PointFeatureExtractor::extract(){
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new  pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(point_ds);
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(searchNormal);
    normalEstimation.compute(*normals);
    edge_points = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    edge_idx.resize(point_ds->size());
    std::iota(edge_idx.begin(), edge_idx.end(), 0);
    edge_points = VoteCriterionExtractor(edge_idx);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_visible = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(auto &point : *point_ds){
        if(point.z < -3.0) continue;
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        point_rgb.r = 255;
        point_rgb.g = 0;
        point_rgb.b = 0;
        point_visible->points.push_back(point_rgb);
    }
    for(auto &point : *edge_points){
        if(point.z < -3.0) continue;
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        point_rgb.r = 80;
        point_rgb.g = 110;
        point_rgb.b = 190;
        point_visible->points.push_back(point_rgb);
    }
}

pcl::PointCloud<PointType>::Ptr PointFeatureExtractor::VoteCriterionExtractor(const vector<int> idxs){
    pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>());
    for(auto i : idxs){
        points->points.push_back(point_ds->points[i]);
    }
    pcl::search::KdTree<PointType>::Ptr KDtree(new pcl::search::KdTree<PointType>);
    KDtree->setInputCloud(points);
    vector<int> neighbors;
    vector<float> distances;
    pcl::PointCloud<PointType>::Ptr edge(new pcl::PointCloud<PointType>);
    edge_idx.clear();
    for (int i = 0; i < points->size(); i++){
        if(KDtree->radiusSearch(points->points[i], 0.1, neighbors, distances) < 50){
            KDtree->nearestKSearch(points->points[i], 50, neighbors, distances);
        }
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int idx : neighbors) {
            centroid += Eigen::Vector3d{points->points[idx].x, points->points[idx].y, points->points[idx].z};
        }
        centroid /= neighbors.size();
        Eigen::Matrix3d cov_matrix = Eigen::Matrix3d::Zero();
        for (int idx = 0; idx < neighbors.size(); ++ idx) {
            Eigen::Vector3d diff = Eigen::Vector3d{points->points[idx].x, points->points[idx].y, points->points[idx].z} - centroid;
            cov_matrix += diff * diff.transpose();
        }
        cov_matrix /= (neighbors.size() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov_matrix);
        Eigen::Vector3d normal_0 = eig.eigenvectors().col(0);
        Eigen::Vector3d normal_1 = eig.eigenvectors().col(1);
        Eigen::Vector3d normal_2 = eig.eigenvectors().col(2);
        Eigen::Vector3d eigenvalues = eig.eigenvalues();
        double curvature = eigenvalues(0) / eigenvalues.sum();
        pcl::Normal normal_svd;
        normal_svd.normal_x = normal_0.x();
        normal_svd.normal_y = normal_0.y();
        normal_svd.normal_z = normal_0.z();
        float blank_rate_0 = VoteCriterion(points, normal_svd, neighbors, i, segAngle);
        normal_svd.normal_x = normal_1.x();
        normal_svd.normal_y = normal_1.y();
        normal_svd.normal_z = normal_1.z();
        float blank_rate_1 = VoteCriterion(points, normal_svd, neighbors, i, segAngle);
        normal_svd.normal_x = normal_2.x();
        normal_svd.normal_y = normal_2.y();
        normal_svd.normal_z = normal_2.z();
        float blank_rate_2 = VoteCriterion(points, normal_svd, neighbors, i, segAngle);
//        if(blank_rate_0 > 0.2){
////            if(!a && vote_rate_0 > 0.35 * exp(accumulate(dists_0.begin(), dists_0.end(), 0.0)/dists_0.size())){
//            edge->points.push_back(points->points[i]);
//            edge_idx.push_back(idxs[i]);
//            continue;
//        }
//        if(blank_rate_0 > blankRate * exp(eigenvalues(0) / eigenvalues.sum())){
//            edge->points.push_back(points->points[i]);
//            edge_idx.push_back(idxs[i]);
//        }
//        if(blank_rate_2 > blankRate * exp(1.0-eigenvalues(2) / eigenvalues.sum())){
//            edge->points.push_back(points->points[i]);
//            edge_idx.push_back(idxs[i]);
//        }
        if(blank_rate_0 > blankRate * exp(eigenvalues(0) / eigenvalues.sum()) && blank_rate_2 > blankRate * exp(1.0-eigenvalues(2) / eigenvalues.sum())){
            edge->points.push_back(points->points[i]);
            edge_idx.push_back(idxs[i]);
        }

    }
    return edge;
}
/**
 * Feature Detector
 * @param points
 * @param normal
 * @param neighbors
 * @param index
 * @param segAngle
 * @return
 */
float PointFeatureExtractor::VoteCriterion(const pcl::PointCloud<PointType>::Ptr points,
                    const pcl::Normal normal,
                    vector<int> neighbors,
                    int index,
                    float segAngle)
{
    if(neighbors.size() < 5) return 0.0;
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), index), neighbors.end());
    std::vector<Eigen::Vector3f> projected_points = ProjectPointsToTangentPlane(points, normal, neighbors, index);
    vector<float> angles;
    PointType p = points->points[index];
    Eigen::Vector3f p_0(p.x - projected_points[1].x(),
                        p.y - projected_points[1].y(),
                        p.z - projected_points[1].z());
    angles.push_back(0.0);
    for(int idx = 2; idx < projected_points.size(); ++idx){
        Eigen::Vector3f p_i(p.x - projected_points[idx].x(),
                            p.y - projected_points[idx].y(),
                            p.z - projected_points[idx].z());
        Eigen::Vector3f cro_v = p_0.cross(p_i);
        if(cro_v.z() > 0)
            angles.push_back(acos(max(-1.0f, min(1.0f, p_0.dot(p_i)/(p_0.norm()*p_i.norm())))));
        else
            angles.push_back(2*M_PI - acos(max(-1.0f, min(1.0f, p_0.dot(p_i)/(p_0.norm()*p_i.norm())))));
    }
    segAngle = segAngle / 180.0 * M_PI;
    vector<int> voteAngle(2*M_PI / segAngle + 1, 0);
    vector<int> voteCount(voteAngle.size(), 1);
    for(auto angle : angles){
        voteAngle[static_cast<int>(floor(angle / segAngle))] = 1;
    }
    for(int i = 0; i < voteCount.size(); ++i){
        if(voteAngle[i] != 0)
            voteCount[i] = 0;
        else if(i > 0)
            voteCount[i] += voteCount[i-1];
    }
    for(auto &num : voteCount){
        if(num == 0) break;
        num += voteCount.back();
    }

    double vote_rate = *max_element(voteCount.begin(), voteCount.end());
    return static_cast<double>(vote_rate)/voteCount.size();
}
/**
 * Planar Mapping
 * @param points
 * @param normal
 * @param neighbors
 * @param center_idx
 * @return
 */
vector<Eigen::Vector3f> PointFeatureExtractor::ProjectPointsToTangentPlane(
        const pcl::PointCloud<PointType>::Ptr points,
        const pcl::Normal normal,
        const vector<int> neighbors,
        int center_idx)
{
    std::vector<Eigen::Vector3f> projected_points;
    PointType center = points->points[center_idx];
    pcl::PointCloud<PointType>::Ptr pro_points(new pcl::PointCloud<PointType>);
    for (int idx : neighbors){
        PointType p = points->points[idx];
        float t = (normal.normal_x*center.x+normal.normal_y*center.y+normal.normal_z*center.z
                   - normal.normal_x*p.x-normal.normal_y*p.y-normal.normal_z*p.z)
                  / (normal.normal_x*normal.normal_x+normal.normal_y*normal.normal_y+normal.normal_z*normal.normal_z);
        Eigen::Vector3f p_pro(p.x + normal.normal_x*t, p.y + normal.normal_y*t, p.z + normal.normal_z*t);
        projected_points.push_back(p_pro);
    }
    Eigen::Vector3f n{normal.normal_x, normal.normal_y, normal.normal_z};
    Eigen::Vector3f c{center.x, center.y, center.z};
    return projected_points;
}
/**
 * Point cloud spherical mapping
 * @param cloud
 * @param cloud_projected
 * @param centerX
 * @param centerY
 * @param centerZ
 * @param radius
 */
void PointFeatureExtractor::projectPointCloudToSphere(
        pcl::PointCloud<PointType>::Ptr cloud,
        pcl::PointCloud<PointType>::Ptr cloud_projected,
        float centerX, float centerY, float centerZ, float radius){
    for (size_t i = 0; i < cloud->points.size(); ++i){
        PointType point;
        float distance = cloud->points[i].getVector3fMap().norm();  // 计算点与原点的距离
        if(distance == 0){
            point.x = 0, point.y = 0, point.z = 0;
            continue;
        }
        point.x = (cloud->points[i].x) * radius / distance + centerX;
        point.y = (cloud->points[i].y) * radius / distance + centerY;
        point.z = (cloud->points[i].z) * radius / distance + centerZ;
//        point.rgb = cloud->points[i].rgb;
        cloud_projected->points.push_back(point);
    }
    cloud_projected->width = cloud->width;
    cloud_projected->height = cloud->height;
    cloud_projected->is_dense = cloud->is_dense;
}

vector<bool> PointFeatureExtractor::visiblePoints(const pcl::PointCloud<PointType>::Ptr points, const Eigen::Matrix3d R, const Eigen::Vector3d t){
    pcl::PointCloud<PointType>::Ptr point_copy(new pcl::PointCloud<PointType>);
    *point_copy = *points;
    vector<double> ranges;
    for(auto &point : *point_copy){
        ranges.push_back(sqrt(point.x*point.x + point.y*point.y + point.z*point.z));
    }
    minRange = *min_element(ranges.begin(), ranges.end());
    maxRange = *max_element(ranges.begin(), ranges.end());
    double scale = 1.0 / *max_element(ranges.begin(), ranges.end());
    for(auto &point : *point_copy){
        Eigen::Vector3d pose{point.x, point.y, point.z};
        pose = pose * scale;
        pose = R * pose + t;
        point.x = pose.x();
        point.y = pose.y();
        point.z = pose.z();
    }
    ranges.clear();
    for(auto &point : *point_copy){
        ranges.push_back(sqrt(point.x*point.x + point.y*point.y + point.z*point.z));
    }
    pcl::PointCloud<PointType>::Ptr points_pro = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    projectPointCloudToSphere(point_copy, points_pro, 0.0, 0.0, 0.0, max(1.0, 2*t.norm()));
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(points_pro);
    pcl::PointCloud<PointType>::Ptr points_vis = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = 0.05;
    vector<bool> visible(point_copy->size(), false);
    double range_dist = (*max_element(ranges.begin(), ranges.end()) - *min_element(ranges.begin(), ranges.end()));
#pragma omp parallel for reduction(+:sum)
    for(int idx = 0; idx < point_copy->size(); ++ idx){
//        kdtree.radiusSearch(points_pro->points[idx], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        kdtree.nearestKSearch(points_pro->points[idx], searchK, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        vector<pair<double, int>> select_points;
        for(auto select_idx : pointIdxRadiusSearch) {
            select_points.push_back(make_pair(ranges[select_idx], select_idx));
        }
        sort(select_points.begin(), select_points.end());
        Eigen::Vector3d cur_point{point_copy->points[idx].x,
                                  point_copy->points[idx].y,
                                  point_copy->points[idx].z};
        double total_a = 0.0;
        double last_a = 0.0;
        for(auto point_near : select_points){
            if(point_near.second == idx) break;
            double cur_a = (ranges[idx] - ranges[point_near.second]) / range_dist;
            Eigen::Vector3d normal_cur{normals->points[idx].normal_x, normals->points[idx].normal_y, normals->points[idx].normal_z};
            if(std::isfinite(normals->points[point_near.second].normal_x) && std::isfinite(normals->points[idx].normal_x)){
                Eigen::Vector3d normal_near{normals->points[point_near.second].normal_x, normals->points[point_near.second].normal_y, normals->points[point_near.second].normal_z};
                sigma = sigma + (normal_cur.normalized().dot(normal_near.normalized()));
            }
            Eigen::Vector3d near_point{point_copy->points[point_near.second].x,
                                       point_copy->points[point_near.second].y,
                                       point_copy->points[point_near.second].z};
            double dist = cur_point.cross(near_point).norm() / ranges[idx];

            cur_a = cur_a * exp(-0.5 * dist*dist / (sigma*sigma));
            cur_a = cur_a * (1.0 - last_a);
            last_a = cur_a;
            total_a += cur_a;
        }
        if(total_a < visThr){
            visible[idx] = true;
        }
    }
    return visible;
}
/**
 * Point Cloud Visibility Analysis Module
 * @param R
 * @param t
 * @return
 */
vector<bool> PointFeatureExtractor::visiblePoints(const Eigen::Matrix3d R, const Eigen::Vector3d t){
    pcl::PointCloud<PointType>::Ptr point_copy(new pcl::PointCloud<PointType>);
    *point_copy = *point_ds;
    vector<double> ranges;
    for(auto &point : *point_copy){
        ranges.push_back(sqrt(point.x*point.x + point.y*point.y + point.z*point.z));
    }
    minRange = *min_element(ranges.begin(), ranges.end());
    maxRange = *max_element(ranges.begin(), ranges.end());
    double scale = 1.0 / *max_element(ranges.begin(), ranges.end());
    for(auto &point : *point_copy){
        Eigen::Vector3d pose{point.x, point.y, point.z};
        pose = pose * scale;
        pose = R * pose + t;
        point.x = pose.x();
        point.y = pose.y();
        point.z = pose.z();
    }
    ranges.clear();
    for(auto &point : *point_copy){
        ranges.push_back(sqrt(point.x*point.x + point.y*point.y + point.z*point.z));
    }
    pcl::PointCloud<PointType>::Ptr points_pro = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    projectPointCloudToSphere(point_copy, points_pro, 0.0, 0.0, 0.0, max(1.0, 2*t.norm()));

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_pro_color = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
//    for(auto &point : *points_pro){
//        pcl::PointXYZRGB point_rgb;
//        point_rgb.x = point.x;
//        point_rgb.y = point.y;
//        point_rgb.z = point.z;
//        point_rgb.r = 80;
//        point_rgb.g = 110;
//        point_rgb.b = 190;
//        point_pro_color->points.push_back(point_rgb);
//    }
//    pcl::io::savePCDFileBinary("point_pro_color.pcd", *point_pro_color);
//    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
//    pcl::PointCloud<pcl::Normal>::Ptr normals(new  pcl::PointCloud<pcl::Normal>);
//    pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
//    normalEstimation.setInputCloud(point_copy);
//    normalEstimation.setSearchMethod(tree);
//    normalEstimation.setRadiusSearch(searchNormal);  // 法向量的半径
//    normalEstimation.compute(*normals);

    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(points_pro);
    pcl::PointCloud<PointType>::Ptr points_vis = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = 0.05;
    vector<bool> visible(point_copy->size(), false);
    double range_dist = (*max_element(ranges.begin(), ranges.end()) - *min_element(ranges.begin(), ranges.end()));
#pragma omp parallel for reduction(+:sum)
    for(int idx = 0; idx < edge_idx.size(); ++ idx){
//        kdtree.radiusSearch(points_pro->points[idx], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        kdtree.nearestKSearch(points_pro->points[edge_idx[idx]], searchK, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        vector<pair<double, int>> select_points;
        for(auto select_idx : pointIdxRadiusSearch) {
            select_points.push_back(make_pair(ranges[select_idx], select_idx));
        }
        sort(select_points.begin(), select_points.end());
        Eigen::Vector3d cur_point{point_copy->points[edge_idx[idx]].x,
                                  point_copy->points[edge_idx[idx]].y,
                                  point_copy->points[edge_idx[idx]].z};
        double total_a = 0.0;
        double last_a = 0.0;
        for(auto point_near : select_points){
            if(point_near.second == edge_idx[idx]) break;
            double cur_a = (ranges[edge_idx[idx]] - ranges[point_near.second]) / range_dist;
            Eigen::Vector3d normal_cur{normals->points[edge_idx[idx]].normal_x, normals->points[edge_idx[idx]].normal_y, normals->points[edge_idx[idx]].normal_z};
            float sig = 0;
            if(std::isfinite(normals->points[point_near.second].normal_x) && std::isfinite(normals->points[edge_idx[idx]].normal_x)){
                Eigen::Vector3d normal_near{normals->points[point_near.second].normal_x, normals->points[point_near.second].normal_y, normals->points[point_near.second].normal_z};
                sig = sigma + (normal_cur.normalized().dot(normal_near.normalized()));
            }
            Eigen::Vector3d near_point{point_copy->points[point_near.second].x,
                                       point_copy->points[point_near.second].y,
                                       point_copy->points[point_near.second].z};
            double dist = cur_point.cross(near_point).norm() / ranges[edge_idx[idx]];
            cur_a = cur_a * exp(-0.5 * dist*dist / (sig*sig));
            cur_a = cur_a * (1.0 - last_a);
            last_a = cur_a;
            total_a += cur_a;
        }
        if(total_a < visThr){
            visible[edge_idx[idx]] = true;
        }
    }
    edge_points->points.clear();
    for(auto idx : edge_idx){
        if(visible[idx]){
            edge_points->points.push_back(point_ds->points[idx]);
        }
    }
    return visible;
}

