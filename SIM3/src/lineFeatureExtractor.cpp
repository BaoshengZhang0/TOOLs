#include "lineFeatureExtractor.h"
void LineFeatureExtractor::LineDetect(const cv::Mat &image, vector<cv::Vec4i> &lines_detected){
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
//    cv::GaussianBlur(grayImage, grayImage, cv::Size(3, 3), 0);
    cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(lengthThr, distanceThr, cannyThr1, cannyThr2, 5);
//    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(10);
    fld->detect(grayImage, lines_detected);
}
/**
 * Depth gradient calculation
 */
void LineFeatureExtractor::SobelLines(){
    Mat gray;
    cvtColor(img_depth, gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat magnitude, angle;
    Sobel(gray, grad_x, CV_64F, 1, 0, 5);
    Sobel(gray, grad_y, CV_64F, 0, 1, 5);

    cartToPolar(grad_x, grad_y, magnitude, angle, false);
    cv::Mat vectorField(angle.size(), CV_32FC2);
    for (int y = 0; y < angle.rows; y++) {
        for (int x = 0; x < angle.cols; x++) {
            float a = angle.at<double>(y, x);
            cv::Point2f vec = cv::Point2f{std::cos(a), std::sin(a)};
            vectorField.at<cv::Vec2f>(y, x) = cv::Vec2f(vec.x, vec.y);
        }
    }
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);
//    cv::imwrite("./sobel.png", magnitude*5);
    vector<vector<cv::Point2i>> linePixels;
    for(auto line : lines_detected_color){
        linePixels.push_back(getLinePixels(line[0], line[1], line[2], line[3]));
        Eigen::Vector2f line_v{line[2]-line[0], line[3]-line[1]};
        double depthGradient = 0.0;
        for(auto p : linePixels.back()){
            float gradient = static_cast<int>(magnitude.at<uchar>(p)) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y}));
            if(p.x-1 > 0){
                gradient = max(gradient, static_cast<int>(magnitude.at<uchar>(Point2i(p.x-1, p.y))) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y})));
            }
            if(p.x+1 < magnitude.cols){
                gradient = max(gradient, static_cast<int>(magnitude.at<uchar>(Point2i(p.x+1, p.y))) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y})));
            }
            if(p.y-1 > 0){
                gradient = max(gradient, static_cast<int>(magnitude.at<uchar>(Point2i(p.x, p.y-1))) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y})));
            }
            if(p.y+1 < magnitude.rows){
                gradient = max(gradient, static_cast<int>(magnitude.at<uchar>(Point2i(p.x, p.y+1))) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y})));
            }
            depthGradient += gradient;
//            depthGradient += static_cast<int>(magnitude.at<uchar>(p)) * abs(line_v.normalized().dot(Eigen::Vector2f{vectorField.at<cv::Point2f>(p).x, vectorField.at<cv::Point2f>(p).y}));
        }
        if(depthGradient / linePixels.back().size() > depthGradientThr/line_v.norm()){
            lines_extracted.push_back(line);
        }
    }
    for(auto line : lines_detected_depth){
        lines_extracted.push_back(line);
    }
}
/**
 * Line Feature Merging
 * @param lines_detected
 */
void LineFeatureExtractor::LineMergeFilter(vector<cv::Vec4i> &lines_detected){
    vector<vector<cv::Vec4i>> lines_jointed;
    for(int i = 0; i < lines_detected.size(); ++ i){
        lines_jointed.push_back(vector<cv::Vec4i>{lines_detected[i]});
        float A = lines_detected[i](3) - lines_detected[i](1);
        float B = lines_detected[i](0) - lines_detected[i](2);
        float C = lines_detected[i](2)*lines_detected[i](1) - lines_detected[i](0)*lines_detected[i](3);
        float len1 = sqrt(pow(lines_detected[i](0)-lines_detected[i](2), 2)+pow(lines_detected[i](1)-lines_detected[i](3), 2));
        for(int j = i+1; j < lines_detected.size(); ++ j){
            float d1 = abs(A * lines_detected[j](0) + B * lines_detected[j](1) + C) / sqrt(A*A + B*B);
            float d2 = abs(A * lines_detected[j](2) + B * lines_detected[j](3) + C) / sqrt(A*A + B*B);

            if(d1 < 3 && d2 < 3){
                float len2 = sqrt(pow(lines_detected[j](0)-lines_detected[j](2), 2)+pow(lines_detected[j](1)-lines_detected[j](3), 2));
                float point_d1 = sqrt(pow(lines_detected[i](0)-lines_detected[j](0), 2)+pow(lines_detected[i](1)-lines_detected[j](1), 2));
                float point_d2 = sqrt(pow(lines_detected[i](0)-lines_detected[j](2), 2)+pow(lines_detected[i](1)-lines_detected[j](3), 2));
                float point_d3 = sqrt(pow(lines_detected[i](2)-lines_detected[j](0), 2)+pow(lines_detected[i](3)-lines_detected[j](1), 2));
                float point_d4 = sqrt(pow(lines_detected[i](2)-lines_detected[j](2), 2)+pow(lines_detected[i](3)-lines_detected[j](3), 2));
                float max_dis = max(max(point_d1, point_d2), max(point_d3, point_d4));
                if(max_dis > (len1+len2) && abs(max_dis-(len1+len2))/max_dis < 0.3){
                    lines_jointed.back().push_back(lines_detected[j]);
                    lines_detected.erase(lines_detected.begin()+j);
                    --j;
                }
                //                lines_jointed.back().push_back(lines_detected[j]);
                //                lines_detected.erase(lines_detected.begin()+j);
                //                --j;
            }
        }
    }
    lines_detected.clear();
    for(auto lines : lines_jointed){
        if(lines.size() > 1){
            vector<pair<float, float>> endpoints;
            float len1 = 0.0;
            for(auto line : lines){
                endpoints.push_back(make_pair(line(0), line(1)));
                endpoints.push_back(make_pair(line(2), line(3)));
                len1 += sqrt(pow(line(0)-line(2), 2)+pow(line(1)-line(3), 2));
            }
            if(abs(lines.front()(0) - lines.front()(2)) > abs(lines.front()(1) - lines.front()(3))){
                sort(endpoints.begin(), endpoints.end(), [](const pair<float, float>& a, const pair<float, float>& b) {
                    return a.first < b.first;
                });
                float len2 = sqrt(pow(endpoints.front().first-endpoints.back().first, 2)+pow(endpoints.front().second-endpoints.back().second, 2));
                if(len1/len2 > 0.9)
                    lines_detected.push_back(cv::Vec4i{endpoints.front().first, endpoints.front().second, endpoints.back().first, endpoints.back().second});
            }else{
                sort(endpoints.begin(), endpoints.end(), [](const pair<float, float>& a, const pair<float, float>& b) {
                    return a.second < b.second;
                });
                float len2 = sqrt(pow(endpoints.front().first-endpoints.back().first, 2)+pow(endpoints.front().second-endpoints.back().second, 2));
                if(len1/len2 > 0.9)
                    lines_detected.push_back(cv::Vec4i{endpoints.front().first, endpoints.front().second, endpoints.back().first, endpoints.back().second});
            }
        }else{
            lines_detected.push_back(lines.front());
        }
    }
}
/**
 * Bresenham Algorithm
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @return
 */
std::vector<cv::Point2i> LineFeatureExtractor::getLinePixels(int x1, int y1, int x2, int y2) {
    std::vector<cv::Point2i> points;
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    int err2;
    while (true) {
        points.push_back(cv::Point2i(x1, y1));
        if (x1 == x2 && y1 == y2) break;
        err2 = 2 * err;
        if (err2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (err2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
    return points;
}