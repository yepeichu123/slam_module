#include "FeatureExtraction.h"
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Error input! please go to the source dir and input ./bin/FeatureExtraction ./data/rgb.png." << endl;
        return 1;
    }
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);

    // detect
    string feat_types = "ORB";
    vector<KeyPoint> kpts;
    Mat desp;
    shared_ptr<FeatureExtraction> featExtra(new FeatureExtraction(feat_types, img));
    featExtra->setDivedeImg_(3,4);
    // featExtra->runFeatureExtractNoDiv(kpts, desp);
    featExtra->runFeatureExtractDiv(kpts, desp);

    // draw
    if (kpts.size() == 0) {
        cout << "detect no keypoint!" << endl;
        return 1;
    }

    Mat out_img;
    drawKeypoints(img, kpts, out_img, Scalar::all(-1),  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", out_img);
    waitKey(0);

    string file = "./data/out_" + feat_types + ".png";
    imwrite(file, out_img);
    cout << "Save image!" << endl;

    return 0;
}
