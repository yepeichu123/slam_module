#include "EpipolarSearch.h"
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void FindRelativePose(const cv::Mat &ref_img, const cv::Mat &cur_img, 
    const cv::Mat &K, cv::Mat &R, cv::Mat &t);

int main(int argc, char** argv) {

    if (argc != 3) {
        cout << "Please go to the source dir, input ./bin/epipolarSearch ./data/1.png ./data/2.png." << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);
    Mat K;
    K = (Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // compute relative pose 
    Mat R, t;
    FindRelativePose(img_1, img_2, K, R, t);

    int padding = 4;
    shared_ptr<EpipolarSearch> eps = make_shared<EpipolarSearch>(K);
    eps->SetupImage(img_2, img_1);
    int nrow = img_1.rows;
    int ncol = img_1.cols;
    vector<Point2f> cur_p, ref_p;
    cout << "Enter runEpipolarSearch!" << endl;
    
    for (int r = padding; r < nrow - padding; ++r) {
        for (int c = padding; c < ncol - padding; ++c) {
            cv::Mat ref_pixel, cur_pixel;
            ref_pixel = (Mat_<float>(2,1) << c, r);

            bool flag = eps->RunEpipolarSearch(ref_pixel, cur_pixel, R, t);
            if (flag) {
                ref_p.push_back(Point2f(c,r));
                cur_p.push_back(Point2f(
                    cur_pixel.at<float>(0), cur_pixel.at<float>(1))
                );
            }
        }
    }
    cout << "Finally, we found " << cur_p.size() << " matching points!" << endl;

    return 0;
}

void FindRelativePose(const cv::Mat &ref_img, const cv::Mat &cur_img, 
    const cv::Mat &K, cv::Mat &R, cv::Mat &t) {

    Ptr<ORB> orb_ = ORB::create();
    vector<KeyPoint> ref_kpt, cur_kpt;
    Mat ref_desp, cur_desp;
    orb_->detectAndCompute(ref_img, Mat(), ref_kpt, ref_desp);
    orb_->detectAndCompute(cur_img, Mat(), cur_kpt, cur_desp);

    vector<DMatch> matches;
    Ptr<BFMatcher> bf_ = BFMatcher::create(NORM_HAMMING2);
    bf_->match(ref_desp, cur_desp, matches);

    float min_dist = min_element(matches.begin(), matches.end(), [](DMatch &m1, DMatch &m2){
        return m1.distance < m2.distance;
    })->distance;

    vector<DMatch> good_matches;
    for (auto m : matches) {
        if (m.distance < max(float(30.0), 2*min_dist)) {
            good_matches.push_back(m);
        }
    }

    vector<Point2f> ref_cam, ref_pixel;
    vector<Point2f> cur_cam, cur_pixel;;
    for (int i = 0; i < good_matches.size(); ++i) {
        Point2f p_ref = ref_kpt[good_matches[i].queryIdx].pt;
        Point2f p_cur = cur_kpt[good_matches[i].trainIdx].pt;
        ref_pixel.push_back(p_ref);
        cur_pixel.push_back(p_cur);


        Point2f p_ref_temp, p_cur_temp;
        p_ref_temp.x = (p_ref.x - K.at<float>(0, 2)) / K.at<float>(0, 0);
        p_ref_temp.y = (p_ref.y - K.at<float>(1, 2)) / K.at<float>(1, 1);
        p_cur_temp.x = (p_cur.x - K.at<float>(0, 2)) / K.at<float>(0, 0);
        p_cur_temp.y = (p_cur.y - K.at<float>(1, 2)) / K.at<float>(1, 1);
        ref_cam.push_back(p_ref_temp);
        cur_cam.push_back(p_cur_temp);
    }

    Mat E = findEssentialMat(ref_pixel, cur_pixel, K, 8, 0.999, 1.0);
    recoverPose(E, ref_pixel, cur_pixel, K, R, t);

    // check the pose 
    float error = 0;
    Mat t_mat = (Mat_<float>(3,3) << 0, -t.at<float>(2,0), t.at<float>(1,0), 
                                     t.at<float>(2,0), 0, -t.at<float>(0,0),
                                     -t.at<float>(1,0), t.at<float>(0,0), 0);
    Mat R_mat;
    R.convertTo(R_mat, CV_32F);
    Mat new_E = t_mat * R_mat;

    for (int i = 0; i < ref_cam.size(); ++i) {
        Mat cur_p, ref_p;
        cur_p = (Mat_<float>(3,1) << cur_cam[i].x, cur_cam[i].y, 1);
        ref_p = (Mat_<float>(3,1) << ref_cam[i].x, ref_cam[i].y, 1);
        Mat d = cur_p.t() * new_E * ref_p;
        error += norm(d);
    }
    cout << "average error is " << error / ref_cam.size() << endl;
}