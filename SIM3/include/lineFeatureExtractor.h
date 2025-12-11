#ifndef CALIBATIONTOOL_LINEFEATUREEXTRACTOR_H
#define CALIBATIONTOOL_LINEFEATUREEXTRACTOR_H
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
using namespace std;
using namespace cv;

class LineFeatureExtractor {
public:
    LineFeatureExtractor(){};
    ~LineFeatureExtractor(){};
    void LineDetect(const cv::Mat &image, vector<cv::Vec4i> &lines_detected);
    void SobelLines();
    void LineMergeFilter(vector<cv::Vec4i> &lines_detected);
    std::vector<cv::Point2i> getLinePixels(int x1, int y1, int x2, int y2);
    cv::Mat img_color;
    cv::Mat img_depth;
    vector<cv::Vec4i> lines_detected_color;
    vector<cv::Vec4i> lines_detected_depth;
    vector<cv::Vec4i> lines_extracted;

    int lengthThr = 30;
    float distanceThr = 1.5;
    double cannyThr1 = 50;
    double cannyThr2 = 150;
    float depthGradientThr = 30.0;
private:

};


#endif //CALIBATIONTOOL_LINEFEATUREEXTRACTOR_H
