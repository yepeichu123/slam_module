#include <iostream>
#include <algorithm>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "TriangularPoints.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 3) {
        cout << "Please go to the source dir and input ./bin/triangular ./data/1.png ./data/2.png." << endl;
        return 1;
    }
    // 1. read image and camera matrix
    Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);
    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
    Mat padding = (Mat_<float>(3, 1) << 0,0,0);
    Mat P_K(3, 4, CV_32F);
    hconcat(K, padding, P_K);

    // 2. compute features from two images
    Ptr<ORB> orb_ = ORB::create(500);
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    orb_->detectAndCompute(img_1, Mat(), kpts_1, desp_1);
    orb_->detectAndCompute(img_2, Mat(), kpts_2, desp_2);

    // 3. compute feature matching with two images
    Ptr<BFMatcher> matcher_ = BFMatcher::create(NORM_HAMMING2);
    vector<DMatch> matches;
    matcher_->match(desp_1, desp_2, matches);

    // 4. refine matching and show images
    vector<Point2f> p2d_1, p2d_2;
    vector<DMatch> good_matches;
    double min_dist = min_element(matches.begin(), matches.end(), 
        [](DMatch &matches_1, DMatch &matches_2) {
            return matches_1.distance < matches_2.distance;
        } )->distance;

    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].distance < max(30.0, min_dist * 3)) {
            good_matches.push_back(matches[i]);

            Point2f p1, p2;
            p1 = kpts_1[matches[i].queryIdx].pt;
            p2 = kpts_2[matches[i].trainIdx].pt;
            p1.x = (p1.x - K.at<float>(0, 2)) / K.at<float>(0, 0);
            p1.y = (p1.y - K.at<float>(1, 2)) / K.at<float>(1, 1);
            p2.x = (p2.x - K.at<float>(0, 2)) / K.at<float>(0, 0);
            p2.y = (p2.y - K.at<float>(1, 2)) / K.at<float>(1, 1);
            p2d_1.push_back(p1);
            p2d_2.push_back(p2);
        }
    }
    cout << "We have " << good_matches.size() << " matching pairs!" << endl;
    Mat match_img;
    drawMatches(img_1, kpts_1, img_2, kpts_2, good_matches, match_img);
    imshow("match_img", match_img);
    waitKey(0);

    // 5. compute relative pose
    // the pose is transforming from frame2 to frame1
    Mat E = findEssentialMat(p2d_1, p2d_2, K, RANSAC, 0.999, 1.0);
    Mat R, t;
    recoverPose(E, p2d_1, p2d_2, K, R, t);

    // 6. triangular points
    // construct transformation matrix
    Mat T_1 = Mat::eye(4, 4, CV_32F);
    T_1.col(3) = 0;
    Mat T_2 = Mat::eye(4, 4, CV_32F);
    Mat T_2_temp_1(3, 4, CV_32F);
    hconcat(R.rowRange(0,3).colRange(0,3), t.rowRange(0,3), T_2_temp_1);
    T_2_temp_1.convertTo(T_2_temp_1, CV_32F);
    Mat T_2_temp_2 = Mat::zeros(1, 4, CV_32F);
    T_2_temp_2.col(3) = 1;
    vconcat(T_2_temp_1, T_2_temp_2, T_2);
    cout << "T_1 = " << T_1 << "\n T_2 = " << T_2 << endl;

    // construct project matrix
    Mat P_1(3, 4, CV_32F), P_2(3, 4, CV_32F);
    P_1 = P_K * T_1;
    P_2 = P_K * T_2;
    cout << "P_1 = " << P_1 << "\n P_2 = " << P_2 << endl;
    // triangulate points
    vector<Point3f> points3d;
    shared_ptr<TriangularPoints> tri_pt = make_shared<TriangularPoints>(K);
    cout << "Enter triangulate points!" << endl;
    // TYPE::ORBSLAM,
    // TYPE::VINS_MONO,
    // TYPE::CV,
    // TYPE::DEFAULT
    // proj_pose is true means we input P = K * T
    // proj_pose is false means we input P = T
    tri_pt->runTriangularPoints(TYPE::ORBSLAM, false, T_1, T_2, kpts_1, kpts_2, good_matches, points3d);
    // tri_pt->runTriangularPoints(TYPE::DEFAULT, true, P_1, P_2, kpts_1, kpts_2, good_matches, points3d);

    // 7. normalize all 3d points 
    int count = 0;
    float sum_depth = 0.0;
    for (int i = 0; i < points3d.size(); ++i) {
        float d = points3d[i].z;

        if (d < 0 || d > 1e+5) {
            continue;
        }
        // calculate all points' depth
        sum_depth += d;
        ++count;
    }
    cout << "we have  = " << count << " 3d points!" << endl;

    float scale = sum_depth / count;
    cout << "rescale = " << scale << endl;
    vector<Point3f> p3d_rescale;
    for (int i = 0; i < points3d.size(); ++i) {
        Point3f p = points3d[i];
        p.z /= scale;
        
        // 8. output depth
        cout << "point " << i << ": normalized depth = " << p.z << endl;
        p3d_rescale.push_back(p);
    }

    return 0;
}