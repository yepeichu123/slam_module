#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "PNP.h"

using namespace cv;
using namespace std;

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp);

void FeatureMatching(Mat &ref_desp, Mat &cur_desp, vector<DMatch> &matches);

void TransformPoints(const Mat &ref_depth, const vector<DMatch> &matches, const Mat &K, const float &scale,
    vector<KeyPoint> &ref_kpts, vector<Point3f> &ref_p3d,
    vector<KeyPoint> &cur_kpts, vector<Point2f> &cur_p2d);


int main(int argc, char** argv) {

    if (argc != 4) {
        cout << "./bin/PNP ./data/ref_rgb.png ./data/ref_depth.png ./data/cur_rgb.png." << endl;
        return -1;
    }

    // read images 
    Mat ref_rgb = imread(argv[1], IMREAD_COLOR);
    Mat ref_depth = imread(argv[2], IMREAD_GRAYSCALE);
    Mat cur_rgb = imread(argv[3], IMREAD_COLOR);
    
    // read camera intrinsics 
    float scale = 5000.0;
    Mat K = (Mat_<float>(3, 3) << 525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1);

    // feature extraction
    vector<KeyPoint> ref_kpts, cur_kpts;
    Mat ref_desp, cur_desp;
    FeatureExtraction(ref_rgb, ref_kpts, ref_desp);
    FeatureExtraction(cur_rgb, cur_kpts, cur_desp);

    // feature matching 
    vector<DMatch> matches;
    FeatureMatching(ref_desp, cur_desp, matches);

    // read depth from depth image 
    vector<Point3f> ref_p3d;
    vector<Point2f> cur_p2d;
    TransformPoints(ref_depth, matches, K, scale, ref_kpts, ref_p3d, cur_kpts, cur_p2d);

    // compute relative pose by pnp 
    PNP my_pnp(K);
    Mat R, t;
    SolvePnpType type = SolvePnpType::PNP_COMB;
    bool flag = my_pnp.RunPNP(ref_p3d, cur_p2d, R, t, type);
    if (flag) {
        cout << "OK, we find the relative pose by pnp. R = \n" << R << "\n t = " << t << endl;
    }
    else {
        cout << "Sorry, we cannot find the relative pose! Please check the wrong message." << endl;
    }

    return 0;
}

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp) {
    Ptr<ORB> orb_ = ORB::create();
    orb_->detectAndCompute(img, Mat(), kpts, desp);
}

void FeatureMatching(Mat &ref_desp, Mat &cur_desp, vector<DMatch> &matches) {
    if (ref_desp.rows == cur_desp.rows && ref_desp.rows > 0 && cur_desp.rows > 0) {
        Ptr<BFMatcher> bf_ = BFMatcher::create(NORM_HAMMING2);
        vector<DMatch> temp_matches;
        bf_->match(ref_desp, cur_desp, temp_matches);

        float min_dist = min_element(temp_matches.begin(), temp_matches.end(), 
            [](DMatch &m1, DMatch &m2) {
                return m1.distance < m2.distance;
            })->distance;
        matches.clear();
        for (auto m : temp_matches) {
            if (m.distance < max(30.0f, min_dist*2)) {
                matches.push_back(m);
            }
        }
        cout << "We have " << matches.size() << " maching pairs!" << endl;
    }
}

void TransformPoints(const Mat &ref_depth, const vector<DMatch> &matches, const Mat &K, const float &scale,
    vector<KeyPoint> &ref_kpts, vector<Point3f> &ref_p3d,
    vector<KeyPoint> &cur_kpts, vector<Point2f> &cur_p2d) {

    if (ref_depth.empty() || matches.size() <= 0 ) {
        cout << "invalid input, please check it again!" << endl;
        return;
    }

    ref_p3d.clear();
    cur_p2d.clear();
    for (int i = 0; i < matches.size(); ++i) {
        Point2f ref_pixel = ref_kpts[matches[i].queryIdx].pt;
        Point2f cur_pixel = cur_kpts[matches[i].trainIdx].pt;

        ushort d = ref_depth.at<ushort>(ref_pixel.y, ref_pixel.x);
        if (d == 0) {
            continue;
        }
        float z = (float)d / scale;
        Point3f ref_p(
            (ref_pixel.x - K.at<float>(0, 2)) * z / K.at<float>(0, 0),
            (ref_pixel.y - K.at<float>(1, 2)) * z / K.at<float>(1, 1),
            z
        );
        ref_p3d.push_back(ref_p);
        cur_p2d.push_back(cur_pixel);
    }
    cout << "Finally, we get " << ref_p3d.size() << " valid points." << endl;
}