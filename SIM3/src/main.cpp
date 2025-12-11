#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <numeric>

// 定义Huber损失函数（鲁棒损失，降低异常值影响）
double huberLoss(double residual, double delta = 1.0) {
    if (std::abs(residual) <= delta) {
        return 0.5 * residual * residual;
    } else {
        return delta * (std::abs(residual) - 0.5 * delta);
    }
}

// 计算单点点云残差（变换后源点与目标点的欧式距离）
double computePointResidual(
        const pcl::PointXYZ& src_pt,
        const pcl::PointXYZ& tgt_pt,
        double s,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t
) {
    Eigen::Vector3d src_vec(src_pt.x, src_pt.y, src_pt.z);
    Eigen::Vector3d tgt_vec(tgt_pt.x, tgt_pt.y, tgt_pt.z);
    // 应用SIM3变换：target = s*R*source + t
    Eigen::Vector3d transformed_src = s * R * src_vec + t;
    // 计算欧式距离作为残差
    return (transformed_src - tgt_vec).norm();
}

// 计算整体残差统计（均值、均方根、中位数）
void computeResidualStats(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
        double s,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        std::vector<double>& residuals,  // 输出每个点的残差
        double& mean_residual,           // 输出均值残差
        double& rmse,                    // 输出均方根误差
        double& median_residual          // 输出中位数残差
) {
    residuals.clear();
    residuals.reserve(source_cloud->size());

    // 计算每个点的残差
    for (size_t i = 0; i < source_cloud->size(); ++i) {
        double res = computePointResidual(source_cloud->points[i], target_cloud->points[i], s, R, t);
        residuals.push_back(res);
    }

    // 计算均值残差
    mean_residual = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();

    // 计算均方根误差（RMSE）
    double sum_sq_res = 0.0;
    for (double res : residuals) {
        sum_sq_res += res * res;
    }
    rmse = std::sqrt(sum_sq_res / residuals.size());

    // 计算中位数残差
    std::vector<double> residuals_sorted = residuals;
    std::sort(residuals_sorted.begin(), residuals_sorted.end());
    size_t mid = residuals_sorted.size() / 2;
    if (residuals_sorted.size() % 2 == 0) {
        median_residual = (residuals_sorted[mid-1] + residuals_sorted[mid]) / 2.0;
    } else {
        median_residual = residuals_sorted[mid];
    }
}

// 基础SIM3估计（单次计算，无迭代）
void estimateSIM3Single(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
        const std::vector<double>& weights,
        double& s,
        Eigen::Matrix3d& R,
        Eigen::Vector3d& t
) {
    if (source_cloud->size() != target_cloud->size() || source_cloud->size() != weights.size() || source_cloud->empty()) {
        throw std::invalid_argument("Point clouds and weights must have same non-empty size");
    }

    // 1. 计算加权质心
    Eigen::Vector3d mu_src(0, 0, 0), mu_tgt(0, 0, 0);
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_weight < 1e-9) {
        throw std::runtime_error("Total weight is too small");
    }

    for (size_t i = 0; i < source_cloud->size(); ++i) {
        const auto& src_pt = source_cloud->points[i];
        const auto& tgt_pt = target_cloud->points[i];
        double w = weights[i];
        mu_src += w * Eigen::Vector3d(src_pt.x, src_pt.y, src_pt.z);
        mu_tgt += w * Eigen::Vector3d(tgt_pt.x, tgt_pt.y, tgt_pt.z);
    }
    mu_src /= total_weight;
    mu_tgt /= total_weight;

    // 2. 中心化点云并计算协方差矩阵H
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    double src_sq_sum = 0.0, tgt_sq_sum = 0.0;
    for (size_t i = 0; i < source_cloud->size(); ++i) {
        const auto& src_pt = source_cloud->points[i];
        const auto& tgt_pt = target_cloud->points[i];
        double w = weights[i];

        Eigen::Vector3d src_centered(src_pt.x - mu_src[0], src_pt.y - mu_src[1], src_pt.z - mu_src[2]);
        Eigen::Vector3d tgt_centered(tgt_pt.x - mu_tgt[0], tgt_pt.y - mu_tgt[1], tgt_pt.z - mu_tgt[2]);

        H += w * tgt_centered * src_centered.transpose();
        src_sq_sum += w * src_centered.squaredNorm();
        tgt_sq_sum += w * tgt_centered.squaredNorm();
    }

    // 3. 计算尺度因子
    s = std::sqrt(tgt_sq_sum / src_sq_sum);

    // 4. SVD分解求旋转矩阵
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    R = U * V.transpose();

    // 确保旋转矩阵行列式为1（避免反射）
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = U * V.transpose();
    }

    // 5. 计算平移向量
    t = mu_tgt - s * R * mu_src;
}

// 迭代鲁棒SIM3估计（IRLS + Huber损失）
void robustEstimateSIM3(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
        std::vector<double>& weights,  // 输入初始权重，输出迭代后权重
        double& s,
        Eigen::Matrix3d& R,
        Eigen::Vector3d& t,
        int max_iter = 2000,             // 最大迭代次数
        double tol = 1e-6,             // 收敛阈值
        double huber_delta = 1.0       // Huber损失阈值
) {
    if (weights.empty()) {
        weights.resize(source_cloud->size(), 1.0);  // 初始权重设为1
    }

    double prev_rmse = 1e9;
    std::vector<double> residuals;
    double mean_residual, rmse, median_residual;

    // 迭代优化
    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. 单次SIM3估计
        estimateSIM3Single(source_cloud, target_cloud, weights, s, R, t);

        // 2. 计算当前残差
        computeResidualStats(source_cloud, target_cloud, s, R, t, residuals, mean_residual, rmse, median_residual);

        // 3. 输出迭代信息
        std::cout << "Iteration " << iter+1 << ":" << std::endl;
        std::cout << "  Mean residual: " << mean_residual << std::endl;
        std::cout << "  RMSE: " << rmse << std::endl;
        std::cout << "  Median residual: " << median_residual << std::endl;

        // 4. 收敛判断（RMSE变化小于阈值）
        if (std::abs(rmse - prev_rmse) < tol) {
            std::cout << "Converged at iteration " << iter+1 << std::endl;
            break;
        }
        prev_rmse = rmse;

        // 5. 更新权重（基于Huber损失的梯度）
        for (size_t i = 0; i < residuals.size(); ++i) {
            double res = residuals[i];
            // Huber损失的权重更新规则：残差越小，权重越高
            if (std::abs(res) <= huber_delta) {
                weights[i] = 1.0;  // 小残差，权重保持1
            } else {
                weights[i] = huber_delta / std::abs(res);  // 大残差，降低权重
            }
        }
    }
}

int main(int argc, char** argv) {
//    if (argc != 3) {
//        std::cerr << "Usage: " << argv[0] << " source.pcd target.pcd" << std::endl;
//        return -1;
//    }

    // 1. 读取PCD点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::string source_pcd_path = "../data/source.pcd";
    std::string target_pcd_path = "../data/target.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_pcd_path, *source_cloud) == -1) {
        std::cerr << "Failed to read source PCD: " << source_pcd_path << std::endl;
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_pcd_path, *target_cloud) == -1) {
        std::cerr << "Failed to read target PCD: " << target_pcd_path << std::endl;
        return -1;
    }

    // 2. 初始化权重（全1）
    std::vector<double> weights(source_cloud->size(), 1.0);

    // 3. 迭代鲁棒估计SIM3变换
    double s;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    try {
        robustEstimateSIM3(source_cloud, target_cloud, weights, s, R, t);
    } catch (const std::exception& e) {
        std::cerr << "Error in SIM3 estimation: " << e.what() << std::endl;
        return -1;
    }

    // 4. 计算最终残差
    std::vector<double> final_residuals;
    double final_mean, final_rmse, final_median;
    computeResidualStats(source_cloud, target_cloud, s, R, t, final_residuals, final_mean, final_rmse, final_median);

    // 5. 输出最终结果
    std::cout << "\n================ Final SIM3 Result ================" << std::endl;
    std::cout << "Scale (s): " << s << std::endl;
    std::cout << "Rotation matrix (R):\n" << R << std::endl;
    std::cout << "Translation vector (t): " << t.transpose() << std::endl;
    std::cout << "Final residual stats:" << std::endl;
    std::cout << "  Mean residual: " << final_mean << std::endl;
    std::cout << "  RMSE: " << final_rmse << std::endl;
    std::cout << "  Median residual: " << final_median << std::endl;

    return 0;
}
