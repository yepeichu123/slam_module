#include "Points.h"
#include "DepthFilter.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

#define IMG_NUM 4

int main(int argc, char** argv) {
    
    if (argc != 2) {
        cout << "Please go to the source dir, and input ./bin/DepthFilter ./data/" << endl;
        return 1;
    }

    // read images
    string in_path = argv[1];
    vector<Mat> imgs;
    for (int i = 0; i < IMG_NUM; ++i) {
        stringstream ss;
        ss << in_path << i << ".png";
        string in_file;
        ss >> in_file;
        Mat img = imread(in_file, IMREAD_GRAYSCALE);
        imgs.push_back(img);
    }
    cout << "We read " << imgs.size() << " images!" << endl;
    if (imgs.size() == 0) {
        cout << "empty images!" << endl;
        return 1;
    }
    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // initialized

    // feature extraction
    vector<Points> my_points_1, my_points_2;
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    Mat first_img = imgs[0];
    Mat second_img = imgs[1];
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(first_img, Mat(), kpts_1, desp_1);
    orb->detectAndCompute(second_img, Mat(), kpts_2, desp_2);

    // feature matching
    vector<DMatch> matches, good_matches;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING2);
    matcher->match(desp_1, desp_2, matches);
    float min_dist = min_element(matches.begin(), matches.end(), 
        [](DMatch &m1, DMatch &m2) {
            return m1.distance < m2.distance;
        }
    )->distance;
    vector<Point2f> pixel_1, pixel_2;
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].distance < max((float)30, min_dist*2)) {
            good_matches.push_back(matches[i]);

            int id_1 = matches[i].queryIdx;
            int id_2 = matches[i].trainIdx;
            Points p1(id_1, K, kpts_1[id_1], desp_1.row(id_1), 0);
            Points p2(id_2, K, kpts_2[id_2], desp_2.row(id_2), 0);
            my_points_1.push_back(p1);
            my_points_2.push_back(p2);

            pixel_1.push_back(kpts_1[id_1].pt);
            pixel_2.push_back(kpts_2[id_2].pt);
        }
    }
    Mat matching_img;
    drawMatches(first_img, kpts_1, second_img, kpts_2, good_matches, matching_img);
    imshow("matching_img", matching_img);
    waitKey(0);

    // compute relative pose 
    Mat r_vec, t_vec;
    Mat E = findEssentialMat(pixel_2, pixel_1, K);
    recoverPose(E, pixel_2, pixel_1, K, r_vec, t_vec);
    Mat R, t;
    r_vec.convertTo(R, CV_32F);
    t_vec.convertTo(t, CV_32F);
    cout << "R = " << R << ", t = " << t << endl;

    // triangulate points 
    vector<Points> new_points;
    DepthFilter d_filter(K);
    for (int i = 0; i < my_points_1.size(); ++i) {
        Points &p1 = my_points_1[i];
        Points &p2 = my_points_2[i];

        d_filter.ComputeTriangulatePoint(R, t, p1, p2);
        d_filter.ComputeUncertainty(R, t, p1, p2);

        if (p1.GetDepth() > 0) {
            new_points.push_back(my_points_1[i]);
        }
    }

    // update points' depth
    for (int i = 2; i < IMG_NUM; ++i) {
        Mat ref_img = imgs[i];
    }

    return 0;
}