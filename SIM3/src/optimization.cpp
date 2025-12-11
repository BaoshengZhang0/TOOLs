#include "optimization.h"
cv::Point2d Optimization::ProjectPointToImage(const PointType& point, const Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    Eigen::Vector3d pointHomogeneous(point.x, point.y, point.z);
    pointHomogeneous = R * pointHomogeneous + t;
    Eigen::Vector3d pointProjected = K * pointHomogeneous;
    pointProjected /= pointProjected.z();
    cv::Point2d pointPixel;
    pointPixel.x = pointProjected.x();
    pointPixel.y = pointProjected.y();
    return pointPixel;
}
void Optimization::OptimizationCeres(){
    Eigen::Quaterniond Qq(RInit);
    double transformInc[7] = {Qq.w(), Qq.x(), Qq.y(), Qq.z(), tInit.x(), tInit.y(), tInit.z()};
    vector<float> max_range(PCF->edge_points->size(), 0.0);
    for(int i = 0; i < PCF->edge_points->size(); ++ i){
        pcl::PointCloud<PointType>::Ptr line_points(new pcl::PointCloud<PointType>);
        for(auto point : *PCF->edge_points){
            float range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            if(range > max_range[i]){
                max_range[i] = range;
            }
        }
    }
    for(int iterCount = 0; iterCount < 30; ++iterCount) {
        ceres::LossFunction *lossFunction = new ceres::HuberLoss(5.0);
        ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
        ceres::Problem problem;
        problem.AddParameterBlock(transformInc, 4, quatParameterization);
        Eigen::Quaterniond q_il{transformInc[0], transformInc[1], transformInc[2], transformInc[3]};
        Eigen::Matrix<double, 3, 1> t_il{transformInc[4], transformInc[5], transformInc[6]};
        PCF->visiblePoints(ROut, tOut);
//        cout << "Visible Point : " << PCF->edge_points->size() << endl;
        double residual_error = 0.0;
        int matched_count = 0;
        for(auto point : *PCF->edge_points){
            Eigen::Matrix<double, 3, 1> ip;
            Eigen::Matrix<double, 3, 1> point_l{point.x, point.y, point.z};
            ip = q_il * point_l + t_il;
            Eigen::Matrix<double, 3, 1> pointProjected = K * ip;
            pointProjected /= pointProjected.z();
            for(auto line : LF->lines_extracted){
                float d1 = sqrt(pow((line(0) - pointProjected.x()), 2) + pow((line(1) - pointProjected.y()), 2));
                float d2 = sqrt(pow((line(2) - pointProjected.x()), 2) + pow((line(3) - pointProjected.y()), 2));
                float len = sqrt(pow((line(0)-line(2)), 2)+pow((line(1)-line(3)), 2));
                if((d1+d2)-len < matchingPara){
                    float A = line(3) - line(1);
                    float B = line(0) - line(2);
                    float C = line(2)*line(1) - line(0)*line(3);
                    residual_error += abs(A*pointProjected.x() + B*pointProjected.y() + C) / sqrt(A*A+B*B);
                    matched_count ++;
                    ceres::CostFunction *costFunction = LinePointFactor::Create(Eigen::Vector3d{A, B, C}, Eigen::Vector3d{point.x, point.y, point.z}, K, exp((d1+d2)/len-1.0), 1.0);
                    problem.AddResidualBlock(costFunction, lossFunction, transformInc, transformInc + 4);
                }
            }
        }
        ceres::Solver::Options solverOptions;
        solverOptions.linear_solver_type = ceres::DENSE_QR;//ceres::SPARSE_SCHUR;
        solverOptions.max_num_iterations = 10;
        solverOptions.max_linear_solver_iterations = 20;
        solverOptions.max_solver_time_in_seconds = 1.0;
        solverOptions.num_threads = 3;
        solverOptions.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        solverOptions.minimizer_progress_to_stdout = false;
        solverOptions.check_gradients = false;
        solverOptions.gradient_check_relative_precision = 1e-6;
        ceres::Solver::Summary summary;
        ceres::Solve(solverOptions, &problem, &summary);
        if (transformInc[0] < 0) {
            Eigen::Quaterniond tmpQ(transformInc[0],
                                    transformInc[1],
                                    transformInc[2],
                                    transformInc[3]);
            if (tmpQ.w() < 0) {
                Eigen::Quaternion<double> resultQ(-tmpQ.w(), -tmpQ.x(), -tmpQ.y(), -tmpQ.z());
                tmpQ = resultQ;
            }
            transformInc[0] = tmpQ.w();
            transformInc[1] = tmpQ.x();
            transformInc[2] = tmpQ.y();
            transformInc[3] = tmpQ.z();
        }
        ROut = Eigen::Quaterniond(transformInc[0], transformInc[1], transformInc[2], transformInc[3]).normalized().toRotationMatrix();
        tOut = Eigen::Vector3d{transformInc[4], transformInc[5], transformInc[6]};
    }
    cout << endl;
    cout << "R:" << endl;
    cout << "[" << ROut(0, 0) << " , " << ROut(0, 1) << " , " << ROut(0, 2) << "]" << endl;
    cout << "[" << ROut(1, 0) << " , " << ROut(1, 1) << " , " << ROut(1, 2) << "]" << endl;
    cout << "[" << ROut(2, 0) << " , " << ROut(2, 1) << " , " << ROut(2, 2) << "]" << endl;
    cout << "t:" << endl;
    cout << "[" << tOut[0] << ", " << tOut[1] << ", " << tOut[2] << "]" << endl;
}