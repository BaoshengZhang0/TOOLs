#include "calibration.h"
void Calibration::ParamerInit(const YAML::Node &config){
    ImgColorPath = config["Common"]["ImgColorPath"].as<string>();
    ImgDepthPath = config["Common"]["ImgDepthPath"].as<string>();
    PcdPath = config["Common"]["PcdPath"].as<string>();
    OutPath = config["Common"]["OutPath"].as<string>();
    downSamplingScale = config["PCF_detection"]["downSamplingScale"].as<double>();
    searchNormal = config["PCF_detection"]["searchNormal"].as<double>();
    segAngle = config["PCF_detection"]["segAngle"].as<double>();
    blankRate = config["PCF_detection"]["blankRate"].as<double>();

    lengthThr = config["IMF_detection"]["lengthThr"].as<int>();
    distanceThr = config["IMF_detection"]["distanceThr"].as<float>();
    cannyThr1 = config["IMF_detection"]["cannyThr1"].as<double>();
    cannyThr2 = config["IMF_detection"]["cannyThr2"].as<double>();
    depthGradientThr = config["IMF_detection"]["depthGradientThr"].as<float>();

    searchK = config["Visible_analysis"]["searchK"].as<int>();
    sigma = config["Visible_analysis"]["sigma"].as<float>();
    visThr = config["Visible_analysis"]["visThr"].as<float>();

    matchingPara = config["Optimization"]["matchingPara"].as<float>();

    fx = config["Camera_internal_parameter"]["fx"].as<double>();
    fy = config["Camera_internal_parameter"]["fy"].as<double>();
    cx = config["Camera_internal_parameter"]["cx"].as<double>();
    cy = config["Camera_internal_parameter"]["cy"].as<double>();
    K << fx, 0, cx,
            0, fy, cy,
            0, 0, 1;
    disEnable = config["Distortion_parameter"]["enable"].as<bool>();
    k1 = config["Distortion_parameter"]["k1"].as<double>();
    k2 = config["Distortion_parameter"]["k2"].as<double>();
    k3 = config["Distortion_parameter"]["k3"].as<double>();
    p1 = config["Distortion_parameter"]["p1"].as<double>();
    p2 = config["Distortion_parameter"]["p2"].as<double>();
    YAML::Node R_node = config["Initial_R"];
    for (int i = 0; i < RInit.rows(); ++i) {
        for (int j = 0; j < RInit.cols(); ++j) {
            RInit(i, j) = R_node[i][j].as<double>();
        }
    }
    YAML::Node t_node = config["Initial_t"];
    tInit.x() = t_node[0][0].as<double>();
    tInit.y() = t_node[0][1].as<double>();
    tInit.z() = t_node[0][2].as<double>();
}
/**
 * Point Feature Extraction
 * @param PCF_
 */
void Calibration::SetPCFDetector(PointFeatureExtractor *PCF_){
    PCF = PCF_;
    PCF->pcdPath = PcdPath;
    PCF->downSamplingScale = downSamplingScale;
    PCF->preprocess();
    PCF->searchNormal = searchNormal;
    PCF->RInit = RInit;
    PCF->tInit = tInit;
    PCF->segAngle = segAngle;
    PCF->blankRate = blankRate;
    PCF->searchK = searchK;
    PCF->sigma = sigma;
    PCF->visThr = visThr;
    PCF->extract();
    PCF->edge_points->width = PCF->edge_points->size();
    PCF->edge_points->height = 1;
    if(!PCF->edge_points->empty()){
        pcl::io::savePCDFileASCII(OutPath + "edge_points.pcd", *PCF->edge_points);
    }
}
/**
 * Image Feature Extraction
 * @param LF_
 */
void Calibration::SetLFDetector(LineFeatureExtractor *LF_){
    LF = LF_;
    LF->img_color = cv::imread(ImgColorPath);
    LF->img_depth = cv::imread(ImgDepthPath);
    LF->lengthThr = lengthThr;
    LF->distanceThr = distanceThr;
    LF->cannyThr1 = cannyThr1;
    LF->cannyThr2 = cannyThr2;
    LF->LineDetect(LF->img_color, LF->lines_detected_color);
    LF->LineDetect(LF->img_depth, LF->lines_detected_depth);
    LF->depthGradientThr = depthGradientThr;
    LF->SobelLines();
    LF->LineMergeFilter(LF->lines_extracted);
    cv::Mat draw1;
    LF->img_depth.copyTo(draw1);
    for(auto line : LF->lines_detected_depth){
        uchar r = std::rand() % 256;
        uchar g = std::rand() % 256;
        uchar b = std::rand() % 256;
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        cv::line(draw1, p1, p2, cv::Scalar(r, g, b), 3);
    }
    cv::Mat draw2;
    LF->img_color.copyTo(draw2);
    draw2 = cv::Mat::zeros(draw2.rows, draw2.cols, CV_8UC3);
    for(auto line : LF->lines_extracted){
        uchar r = std::rand() % 256;
        uchar g = std::rand() % 256;
        uchar b = std::rand() % 256;
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        cv::line(draw2, p1, p2, cv::Scalar(r, g, b), 3);
    }
    cv::imwrite(OutPath + "depth_lines.png", draw1);
}
/**
 * System Optimization
 * @param OP_
 */
void Calibration::SetOptimizer(Optimization *OP_){
    OP = OP_;
    OP->PCF = PCF;
    OP->LF = LF;
    OP->RInit = RInit;
    OP->tInit = tInit;
    OP->ROut = RInit;
    OP->tOut = tInit;
    OP->K = K;
    OP->matchingPara = matchingPara;
    OP->OptimizationCeres();
    R = OP->ROut;
    t = OP->tOut;
//    cv::Mat img_display1, img_display2, img_display3;
//    LF->img_color.copyTo(img_display1);
//    LF->img_color.copyTo(img_display2);
//    LF->img_color.copyTo(img_display3);
//    for(auto point : PCF->edge_points->points){
//        cv::Point2i pointPixel = OP->ProjectPointToImage(point, RInit, tInit);
//        if((pointPixel.x >= 0 && pointPixel.x < LF->img_color.cols) && (pointPixel.y >= 0 && pointPixel.y < LF->img_color.rows)){
//            cv::circle(img_display1, Point2d(pointPixel.x, pointPixel.y), 3, Scalar(0, 255, 255), -1);
//        }
//    }
//    for(auto point : PCF->edge_points->points){
//        cv::Point2i pointPixel = OP->ProjectPointToImage(point, OP->ROut, OP->tOut);
//        if((pointPixel.x >= 0 && pointPixel.x < LF->img_color.cols) && (pointPixel.y >= 0 && pointPixel.y < LF->img_color.rows)){
//            cv::circle(img_display2, Point2d(pointPixel.x, pointPixel.y), 3, Scalar(255, 0, 255), -1);
//        }
//    }
}

Eigen::Vector2d Calibration::ProjectPointToImage(const Eigen::Vector3d& point) {
    Eigen::Vector3d pointHomogeneous(point.x(), point.y(), point.z());
    pointHomogeneous = R * pointHomogeneous + t;
    Eigen::Vector3d pointProjected = K * pointHomogeneous;
    pointProjected /= pointProjected.z();
    Eigen::Vector2d pointPixel;
    pointPixel(0) = pointProjected.x();
    pointPixel(1) = pointProjected.y();
    return pointPixel;
}

void Calibration::MapCustomColor(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = 255;
    g = 255;
    b = 255;
    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    double normalized = (vmax == vmin) ? 0.0 : (v - vmin) / (vmax - vmin);

    double dr, dg, db;
    if (normalized < 0.25) {
        dr = 0.3 + normalized * 0.4;
        db = 0.8 - normalized * 0.2;
        dg = 0.0;
    } else if (normalized < 0.5) {
        dr = 0.7 - (normalized - 0.25) * 0.7;
        db = 0.6 + (normalized - 0.25) * 0.4;
        dg = 0.0 + (normalized - 0.25) * 0.2;
    } else if (normalized < 0.75) {
        dr = 0.0;
        db = 1.0 - (normalized - 0.5) * 0.5;
        dg = 0.2 + (normalized - 0.5) * 0.8;
    } else {
        dr = 0.0 + (normalized - 0.75) * 0.2;
        db = 0.5 - (normalized - 0.75) * 0.5;
        dg = 1.0;
    }
    r = static_cast<uint8_t>(255 * dr);
    g = static_cast<uint8_t>(255 * dg);
    b = static_cast<uint8_t>(255 * db);
}

void Calibration::ResultOut(){
    pcl::PointCloud<PointType>::Ptr pl_pcd(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>(PcdPath, *pl_pcd) == -1){
        std::cerr << "Couldn't read PCD file." << std::endl;
        return;
    }
    cv::Mat Img = imread(ImgColorPath);
    if (Img.empty()) {
        std::cout << "Couldn't read Image file." << std::endl;
        return;
    }
    int size = 5;
    int segW = size, segH = size;
    int numW = floor(Img.cols/segW), numH = floor(Img.rows/segH);
    int marW = floor((Img.cols%segW)/2+1), marH = floor((Img.rows%segH)/2+1);
    cv::Mat ImgPro;
    Img.copyTo(ImgPro);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointxyzrgbs(new pcl::PointCloud<pcl::PointXYZRGB>);
    vector<float> ranges;
    for(auto point : pl_pcd->points) {
        if (isnan(point.x) || isnan(point.y) || isnan(point.z)) continue;
        ranges.push_back(sqrt(point.x*point.x + point.y*point.y + point.z*point.z));
    }
    float min_range = *std::min_element(ranges.begin(), ranges.end());
    float max_range = *std::max_element(ranges.begin(), ranges.end());
    for(auto point : pl_pcd->points){
        if(isnan(point.x) || isnan(point.y) || isnan(point.z)) continue;
        if(point.x < 0) continue;
        Eigen::Vector3d points(point.x, point.y, point.z); // 示例点，x, y, z坐标
        double range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
        double inten = point.intensity;
        uint8_t r, g, b;
        MapCustomColor(inten, 0, 255, r, g, b);
        Eigen::Vector2d pointPixel = ProjectPointToImage(points);
        if((pointPixel(0) >= marW && pointPixel(0) < Img.cols-marW) && (pointPixel(1) >= marH && pointPixel(1) < Img.rows-marH)){
            cv::circle(ImgPro, Point2d(pointPixel(0), pointPixel(1)), 1, Scalar(b, g, r), -1);
            pcl::PointXYZRGB pointxyzrgb;
            pointxyzrgb.x = point.x;
            pointxyzrgb.y = point.y;
            pointxyzrgb.z = point.z;
            cv::Vec3b pixel = Img.at<cv::Vec3b>(cv::Point(pointPixel.x(), pointPixel.y()));
            pointxyzrgb.b = pixel[0];
            pointxyzrgb.g = pixel[1];
            pointxyzrgb.r = pixel[2];
            pointxyzrgbs->push_back(pointxyzrgb);
        }
    }
    pcl::io::savePCDFile(OutPath+"colorize_points.pcd", *pointxyzrgbs);
    cv::addWeighted(Img, 0.7, ImgPro, 0.7, 0.0, Img);
    cv::imwrite(OutPath+"calibrated_img.png", Img);
    cv::imshow("calibrated image", Img);
    cv::waitKey(0);
    return;
}