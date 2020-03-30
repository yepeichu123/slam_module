#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "EpipolarConstrain.h"
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
    float min_dist = min_element(temp_matches.begin(), temp_matches.end(), 
        [](DMatch &m1, DMatch &m2) {
            return m1.distance < m2.distance;
        })->distance;
    for (int i = 0; i < temp_matches.size(); ++i) {
        if (temp_matches[i].distance < max(30.f, min_dist)) {
            good_matches.push_back(temp_matches[i]);
        }
    }
    cout << "After filter, we have " << good_matches.size() << " matching pairs!" << endl;
    Mat first_matching;
    drawMatches(img_1, kpts_1, img_2, kpts_2, good_matches, first_matching);
    imshow("first_matching", first_matching);
    waitKey(0);

    EpipolarConstrain my_ec(K);
    Mat R, t;
    my_ec.ComputeRelativePose(kpts_1, kpts_2, good_matches, R, t);
    Mat t_x = (Mat_<float>(3, 3) << 0, -t.at<float>(2,0), t.at<float>(1,0),
                                    t.at<float>(2,0), 0, -t.at<float>(0,0),
                                    -t.at<float>(1,0), t.at<float>(0,0), 0);
    Mat E = t_x * R;
    cout << "E = " << E << endl;

    vector<Point2f> p1, p2;
    if (good_matches.size() > 8) {
        for (int i = 0; i < 8; ++i) {
            p1.push_back(kpts_1[i].pt);
            p2.push_back(kpts_2[i].pt);
        }
        Mat E_new;
        my_ec.EpipolarConstrainFor8Points(p1, p2, E_new);
        cout << "E_new = " << E_new << endl;
    }
    Mat ransac_matching;
    drawMatches(img_1, kpts_1, img_2, kpts_2, good_matches, ransac_matching);
    imshow("ransac_matching", ransac_matching);
    waitKey(0);

    return 0;
}