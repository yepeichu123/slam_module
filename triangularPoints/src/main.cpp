#include <iostream>
#include <algorithm>

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
    
    // 2. compute features from two images
    Ptr<ORB> orb_ = ORB::create();
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
        if (matches[i].distance < min_dist * 2) {

            good_matches.push_back(matches[i]);

            Point2f p1, p2;
            p1 = kpts_1[matches[i].queryIdx].pt;
            p2 = kpts_2[matches[i].trainIdx].pt;

            p1.x = (p1.x - K.at<float>(0, 2)) * K.at<float>(0, 0);
            p1.y = (p1.y - K.at<float>(1, 2)) * K.at<float>(1, 1);
            p2.x = (p2.x - K.at<float>(0, 2)) * K.at<float>(0, 0);
            p2.y = (p2.y - K.at<float>(1, 2)) * K.at<float>(1, 1);

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
    cout << "find essential matrix!" << endl;
    Mat E = findEssentialMat(p2d_1, p2d_2, K, RANSAC, 0.999, 1.0);
    Mat R, t;
    cout << "recover pose!" << endl;
    recoverPose(E, p2d_1, p2d_2, K, R, t);

    // 6. triangular points
    Mat P_1 = (Mat_<float>(3, 4) << 1, 0, 0, 0, 
                                    0, 1, 0, 0,
                                    0, 0, 1, 0);
    Mat P_2 = Mat::zeros(Size(4, 3), CV_32FC1);
    Mat pts_4d;
    hconcat(R.rowRange(0, 3).colRange(0, 3), t.rowRange(0,3), P_2);
    triangulatePoints(P_1, P_2, p2d_1, p2d_2, pts_4d);

    // 7. normalize all 3d points 
    vector<Point3f> p3d;
    int count = 0;
    float sum_depth = 0.0;
    for (int i = 0; i < pts_4d.cols; ++i) {
        Mat points = pts_4d.col(i);
        points = points.rowRange(0, 3) / points.at<float>(3);
        Point3f p(points.at<float>(0), points.at<float>(1), points.at<float>(2));
        p3d.push_back(p);

        // calculate all points
        sum_depth += p.z;
        ++count;
    }
    cout << "we have  = " << p3d.size() << " 3d points!" << endl;

    float scale = sum_depth / count;
    cout << "rescale = " << scale << endl;
    vector<Point3f> p3d_rescale;
    for (int i = 0; i < p3d.size(); ++i) {
        Point3f p = p3d[i];
        p.z *= scale;
        
        // 8. output depth
        cout << "point " << i << ": normalized depth = " << p.z << endl;
        p3d_rescale.push_back(p);
    }


    return 0;
}