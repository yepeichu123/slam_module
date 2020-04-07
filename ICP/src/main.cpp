#include <iostream>
#include "ICP.h"
#include "ICP_G2O.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void FeatureExtraction(const Mat& img, vector<KeyPoint>& kpts, Mat& desp);

void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<DMatch>& matches);

void TransformPoints3d(const Mat& K, const Mat& ref_depth, const Mat& cur_depth, float& scale, vector<DMatch> &matches,
    vector<KeyPoint>& ref_kpts, vector<Point3f>& ref_vp3d,
    vector<KeyPoint>& cur_kpts, vector<Point3f>& cur_vp3d);

int main(int argc, char** argv) {

    if (argc != 5) {
        cout << "Please input ./bin/ICP ./data/ref_rgb.png ./data/ref_depth.png ./data/cur_rgb.png ./data/cur_depth.png." << endl;
        return -1;
    }

    // read images 
    Mat ref_rgb = imread(argv[1], IMREAD_COLOR);
    Mat ref_depth = imread(argv[2], IMREAD_GRAYSCALE);
    Mat cur_rgb = imread(argv[3], IMREAD_COLOR);
    Mat cur_depth = imread(argv[4], IMREAD_GRAYSCALE);

    // read camera intrinsics
    float scale = 5000.0;
    Mat K = (Mat_<float>(3, 3) << 525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1);

    // feature extraction and matching 
    vector<KeyPoint> ref_kpts, cur_kpts;
    Mat ref_desp, cur_desp;
    vector<DMatch> matches;
    FeatureExtraction(ref_rgb, ref_kpts, ref_desp);
    FeatureExtraction(cur_rgb, cur_kpts, cur_desp);
    FeatureMatching(ref_desp, cur_desp, matches);

    // transform points to 3d
    vector<Point3f> ref_vp3d, cur_vp3d;
    TransformPoints3d(K, ref_depth, cur_depth, scale, matches, ref_kpts, ref_vp3d, cur_kpts, cur_vp3d);

    ICP my_icp;
    ICP_TYPE type = ICP_TYPE::BA;
    Mat R, t;
    bool flag = my_icp.RunICP(ref_vp3d, cur_vp3d, R, t, type);
    if (flag) {
        cout << "We compute relative pose : \n R = " << R << "\n t = " << t << endl;
    }

    return 0;
}

void FeatureExtraction(const Mat& img, vector<KeyPoint>& kpts, Mat& desp) {
    Ptr<ORB> orb_ = ORB::create();
    orb_->detectAndCompute(img, Mat(), kpts, desp);
    cout << "We extracte " << kpts.size() << " from image!" << endl;
}

void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<DMatch>& matches) {
    if (ref_desp.rows > 0 && cur_desp.rows > 0) {
        vector<DMatch> temp_matches;
        Ptr<BFMatcher> bf_ = BFMatcher::create(NORM_HAMMING2);
        bf_->match(ref_desp, cur_desp, temp_matches);
        cout << "First time, we matching " << temp_matches.size() << " points!" << endl;

        float min_dist = min_element(temp_matches.begin(), temp_matches.end(), 
            [](DMatch &m1, DMatch &m2) {
                return m1.distance < m2.distance;
            })->distance;

        if (temp_matches.size() > 0) {
            matches.clear();
        }
        for (auto m : temp_matches) {
            if (m.distance < max(30.0f, min_dist*2)) {
                matches.push_back(m);
            }
        }
        cout << "After filter, we have " << matches.size() << " matching pairs left." << endl;
    }
}

void TransformPoints3d(const Mat& K, const Mat& ref_depth, const Mat& cur_depth, float& scale, vector<DMatch> &matches,
    vector<KeyPoint>& ref_kpts, vector<Point3f>& ref_vp3d,
    vector<KeyPoint>& cur_kpts, vector<Point3f>& cur_vp3d) {

    if (!ref_depth.empty() && !cur_depth.empty() && matches.size() > 0) {

        ref_vp3d.clear();
        cur_vp3d.clear();

        for (int i = 0; i < matches.size(); ++i) {
            Point2f ref_p = ref_kpts[matches[i].queryIdx].pt;
            Point2f cur_p = cur_kpts[matches[i].trainIdx].pt;
            ushort ref_d = ref_depth.at<ushort>(ref_p.y, ref_p.x);
            ushort cur_d = cur_depth.at<ushort>(cur_p.y, cur_p.x);
            if (ref_d == 0 || cur_d == 0) {
                continue;
            }
            float ref_z = (float)ref_d / scale;
            float cur_z = (float)cur_d / scale;
            Point3f ref_p3d(
                (ref_p.x - K.at<float>(0, 2)) * ref_z / K.at<float>(0, 0),
                (ref_p.y - K.at<float>(1, 2)) * ref_z / K.at<float>(1, 1),
                ref_z
            );
            Point3f cur_p3d(
                (cur_p.x - K.at<float>(0, 2)) * cur_z / K.at<float>(0, 0),
                (cur_p.y - K.at<float>(1, 2)) * cur_z / K.at<float>(1, 1),
                cur_z
            );
            ref_vp3d.push_back(ref_p3d);
            cur_vp3d.push_back(cur_p3d);
        }

        cout << "We convert " << ref_vp3d.size() << " points from mat to point3f." << endl;
    }
}