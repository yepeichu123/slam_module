#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 3) {
        cout << "Please input ./bin/EpipolarConstrain ./data/1.png ./data/2.png." << endl;
        return 1;
    }

    // read images 
    Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (img_1.empty() || img_2.empty()) {
        cout << "Please check valuable image's path." << endl;
        return 1;
    }

    // read camera intrinsics 
    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // feature extraction and feature matching
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    vector<DMatch> temp_matches, good_matches;
    Ptr<ORB> orb_ = ORB::create();
    orb_->detectAndCompute(img_1, Mat(), kpts_1, desp_1);
    orb_->detectAndCompute(img_2, Mat(), kpts_2, desp_2);
    Ptr<BFMatcher> bf_ = BFMatcher::create(NORM_HAMMING2);
    bf_->match(desp_1, desp_2, temp_matches);
    cout << "we find " << temp_matches.size() << " matching points first time!" << endl;

    return 0;
}