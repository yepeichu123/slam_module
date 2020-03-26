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

#define IMG_NUM 6

void ReadImagesFormFiles(string &in_path, vector<Mat> &images);

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp);

void FeatureMatching(Mat &desp_1, Mat &desp_2, vector<DMatch> &matches);

void ComputeRelativePose(vector<Point2f> &pixel_1, vector<Point2f> &pixel_2, const Mat &K,
    Mat &R, Mat &t);

// use current image to triangulate points in reference image
void InitializedPoints(const Mat &K, const Mat &first_img, const Mat &second_img, 
    vector<Points> &ref_points, vector<KeyPoint> &ref_kpts, Mat &ref_desp);

int main(int argc, char** argv) {
    
    if (argc != 2) {
        cout << "Please go to the source dir, and input ./bin/DepthFilter ./data/" << endl;
        return 1;
    }

    // read images
    string in_path = argv[1];
    vector<Mat> imgs;
    ReadImagesFormFiles(in_path, imgs);
    if (imgs.size() == 0) {
        cout << "empty images!" << endl;
        return 1;
    }

    Mat K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // initialized
    Mat first_img = imgs[0];
    Mat second_img = imgs[1];
    vector<KeyPoint> first_kpts;
    Mat first_desp;
    vector<Points> first_points;
    InitializedPoints(K, first_img, second_img, first_points, first_kpts, first_desp);
    for (int i = 0; i < first_points.size(); ++i) {
        cout << first_points[i].GetPointId() << " depth = " << first_points[i].GetDepth() << ", uncertainty = " << first_points[i].GetUncertainty() << endl;
    }

    // update points' depth
    for (int i = 2; i < IMG_NUM; ++i) {
        Mat cur_img = imgs[i];
        Mat cur_desp;
        vector<KeyPoint> cur_kpts;
        vector<DMatch> matches;
        FeatureExtraction(cur_img, cur_kpts, cur_desp);
        FeatureMatching(first_desp, cur_desp, matches);

        vector<Point2f> pixel_1, pixel_2;
        for (int m = 0; m < matches.size(); ++m) {
            pixel_1.push_back(first_kpts[matches[m].queryIdx].pt);
            pixel_2.push_back(cur_kpts[matches[m].trainIdx].pt);
        }

        // compute relative pose from current frame to first frame 
        Mat R, t;
        ComputeRelativePose(pixel_2, pixel_1, K, R, t);

        cout << endl;
        // triangulate points 
        DepthFilter my_DF(K);
        vector<Points> ref_points, cur_points;
        for (int k = 0; k < matches.size(); ++k) {
            int id_1 = matches[k].queryIdx;
            int id_2 = matches[k].trainIdx;

            Points p1(id_1, K, first_kpts[id_1], first_desp.row(id_1), 0);
            Points p2(id_2, K, cur_kpts[id_2], cur_desp.row(id_2), 0);
            my_DF.ComputeTriangulatePoint(R, t, p1, p2);
            if (p1.GetDepth() <= 0) {
                continue;
            }
            my_DF.ComputeUncertainty(R, t, p1, p2);
            ref_points.push_back(p1);
            cur_points.push_back(p2);
        }

        // search points and update depth 
        for (int h = 0; h < ref_points.size(); ++h) {
            int id_1 = -1;
            int index_1 = -1;
            int id_2 = ref_points[h].GetPointId();
            int index_2 = h;
            // search points 
            for (int j = 0; j < first_points.size(); ++j) {
                int id = first_points[j].GetPointId();
                if (id < id_2) {
                    continue;
                }
                else if (id == id_2) {
                    id_1 = id;
                    index_1 = j;
                    break;
                }
                else {
                    break;
                }
            }
            // update points' depth 
            if (index_1 != -1) {
                my_DF.RunSingleDepthFilter(first_points[index_1], ref_points[index_2]);
            }
        }
    }
    cout << "after depth filter!" << endl;
    // after depth filter 
    for (int i = 0; i < first_points.size(); ++i) {
        cout << first_points[i].GetPointId() << " depth = " << first_points[i].GetDepth() << ", uncertainty = " << first_points[i].GetUncertainty() << endl;
    }

    return 0;
}

void ReadImagesFormFiles(string &in_path, vector<Mat> &images) {
    // read images
    for (int i = 0; i < IMG_NUM; ++i) {
        stringstream ss;
        ss << in_path << i << ".png";
        string in_file;
        ss >> in_file;
        Mat img = imread(in_file, IMREAD_GRAYSCALE);
        images.push_back(img);
    }
    cout << "We read " << images.size() << " images!" << endl;
}

void FeatureExtraction(const Mat &img, vector<KeyPoint> &kpts, Mat& desp) {
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(img, Mat(), kpts, desp);
}

void FeatureMatching(Mat &desp_1, Mat &desp_2, vector<DMatch> &matches) {
    // feature matching
    vector<DMatch> temp_matches;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING2);
    matcher->match(desp_1, desp_2, temp_matches);
    float min_dist = min_element(temp_matches.begin(), temp_matches.end(), 
        [](DMatch &m1, DMatch &m2) {
            return m1.distance < m2.distance;
        }
    )->distance;
    for (int i = 0; i < temp_matches.size(); ++i) {
        if (temp_matches[i].distance < max((float)30, min_dist*2)) {
            matches.push_back(temp_matches[i]);
        }
    }
    cout << "Here we found " << matches.size() << " matching pairs!" << endl;
}

void ComputeRelativePose(vector<Point2f> &pixel_1, vector<Point2f> &pixel_2, const Mat &K,
    Mat &R, Mat &t) {
    // compute relative pose 
    Mat r_vec, t_vec;
    // We compute the pose from pixel_1 image frame to pixel_2 image frame
    Mat E = findEssentialMat(pixel_1, pixel_2, K);
    recoverPose(E, pixel_1, pixel_2, K, r_vec, t_vec);
    r_vec.convertTo(R, CV_32F);
    t_vec.convertTo(t, CV_32F);

    Mat tx = (Mat_<float>(3, 3) << 0, -t.at<float>(2,0), t.at<float>(1,0),
                                   t.at<float>(2,0), 0, -t.at<float>(0,0),
                                   -t.at<float>(1,0), t.at<float>(0,0), 0);

    // check relative pose
    vector<Mat> p3d_1, p3d_2;
    float error = 0;
    for (int i = 0; i < pixel_1.size(); ++i) {
        Mat p1 = (Mat_<float>(3, 1) << (pixel_1[i].x - K.at<float>(0, 2)) / K.at<float>(0, 0), 
                                       (pixel_1[i].y - K.at<float>(1, 2)) / K.at<float>(1, 1), 
                                       1);
        Mat p2 = (Mat_<float>(3, 1) << (pixel_2[i].x - K.at<float>(0, 2)) / K.at<float>(0, 0), 
                                       (pixel_2[i].y - K.at<float>(1, 2)) / K.at<float>(1, 1), 
                                       1);
        Mat e = p2.t() * tx * R * p1;
        error += norm(e);
    }
    cout << "average error = " << error / pixel_1.size() << endl;
}   

// use current image to triangulate points in reference image
void InitializedPoints(const Mat &K, const Mat &first_img, const Mat &second_img, 
    vector<Points> &ref_points, vector<KeyPoint> &ref_kpts, Mat &ref_desp) {
    // extract features 
    vector<KeyPoint> kpts_1, kpts_2;
    Mat desp_1, desp_2;
    FeatureExtraction(first_img, kpts_1, desp_1);
    FeatureExtraction(second_img, kpts_2, desp_2);

    // feature matching 
    vector<DMatch> matches;
    FeatureMatching(desp_1, desp_2, matches);

    // compute relative pose from second_img to first_img
    // because we want to use second frame to triangulate the points in first_img
    vector<Point2f> pixel_1, pixel_2;
    vector<Points> my_points_1, my_points_2;
    for (int i = 0; i < matches.size(); ++i) {
        int id_1 = matches[i].queryIdx;
        int id_2 = matches[i].trainIdx;

        pixel_1.push_back(kpts_1[id_1].pt);
        pixel_2.push_back(kpts_2[id_2].pt);

        Points p1(id_1, K, kpts_1[id_1], desp_1.row(id_1), 0);
        Points p2(id_2, K, kpts_2[id_2], desp_2.row(id_2), 0);
        my_points_1.push_back(p1);
        my_points_2.push_back(p2);
    }
    Mat R_12, t_12;
    ComputeRelativePose(pixel_2, pixel_1, K, R_12, t_12);

    // triangulate points   
    DepthFilter my_DF(K);
    for (int i = 0; i < my_points_1.size(); ++i) {
        Points &p1 = my_points_1[i];
        Points &p2 = my_points_2[i];

        my_DF.ComputeTriangulatePoint(R_12, t_12, p1, p2);
        my_DF.ComputeUncertainty(R_12, t_12, p1, p2);
        float d = p1.GetDepth();
        if (d > 0) {
            ref_points.push_back(p1);
        }
    }

    ref_kpts.insert(ref_kpts.end(), kpts_1.begin(), kpts_1.end());
    ref_desp = desp_1.clone();
}