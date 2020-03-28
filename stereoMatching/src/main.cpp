// c++
#include <iostream>
#include <string>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// self defined
#include "StereoMatching.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Please input ./bin/stereoMatching ./data/." << endl;
        return 1;
    }

    string left_file = string(argv[1]) + "left.png";
    string right_file = string(argv[1]) + "right.png";

    // Read images 
    Mat left_img = imread(left_file, IMREAD_GRAYSCALE);
    Mat right_img = imread(right_file, IMREAD_GRAYSCALE);
    if (left_img.empty() || right_img.empty()) {
        cout << "Empty images! Please check your path!" << endl;
        return 1;
    }

    // camera intrinsics
    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // Stereo Matching 
    Mat R = Mat::eye(3, 3, CV_32F);
    Mat t = (Mat_<float>(3, 1) << 0.537150653267924, 0, 0);
    Mat disparity, depth;
    StereoMatching smatch(K, left_img.rows, left_img.cols);
    smatch.RunStereoMatching(left_img, right_img, R, t, disparity, depth);

    string out_file = string(argv[1]) + "disparity.png";
    string depth_file = string(argv[1]) + "depth.png";
    if (!disparity.empty()) {
        imwrite(out_file, disparity);
        imshow("disparity", disparity);
        waitKey(0);
    }
    if (!depth.empty()) {
        imwrite(depth_file, depth);
        imshow("depth", depth);
        waitKey(0);
    }

    /*
    Mat depth_img = imread(depth_file, IMREAD_GRAYSCALE);
    for (int i = 0; i < depth_img.rows; ++i) {
        for (int j = 0; j < depth_img.cols; ++j) {
            cout << depth_img.at<ushort>(i,j) << " ";
        }
        cout << endl;
    }
    */    

    return 0;
}