#include "EpipolarSearch.h"
#include <iostream>

using namespace std;

EpipolarSearch::EpipolarSearch(const cv::Mat &K):
    mmK_(K) {
    
    mnWin_ = 5;
}

EpipolarSearch::EpipolarSearch(const cv::Mat &K, const cv::Mat &ref_img, const cv::Mat &cur_img):
    mmK_(K), mmRefImg_(ref_img), mmCurImg_(cur_img) {

    mnWin_ = 5;
}

EpipolarSearch::~EpipolarSearch() {

}

bool EpipolarSearch::RunEpipolarSearch(const cv::Mat &ref_p, cv::Mat &cur_p,
    const cv::Mat &R, cv::Mat &t) {

    if (mmCurImg_.empty() || mmRefImg_.empty()) {
        cout << "Please setup the current image and reference image." << endl;
        return false;
    }

    cv::Mat ref_p3d(3, 1, CV_32F);
    PixelToCam_(ref_p, ref_p3d);

    // project reference point to current frame 
    float min_depth = 0.1,  max_depth = 100;
    cv::Mat new_R, new_t;
    R.convertTo(new_R, CV_32F);
    t.convertTo(new_t, CV_32F);
    cv::Mat min_cur_p = mmK_ * (min_depth * new_R * ref_p3d) + mmK_ * new_t;
    cv::Mat max_cur_p = mmK_ * (max_depth * new_R * ref_p3d) + mmK_ * new_t;

    // compute epipolar an direction 
    cv::Mat epipolar_line;
    epipolar_line = max_cur_p - min_cur_p;
    float epipolar_length = cv::norm(epipolar_line);
    // cout << "epipolar_length = " << epipolar_length << endl;
    cv::Mat epipolar_direction = epipolar_line;
    cv::normalize(epipolar_line, epipolar_direction, 1, 0);

    cv::Mat best_cur_p;
    float best_ncc = 0;
    for (float l = 0; l <= epipolar_length; l+=1.414) {
        cv::Mat px_cur = min_cur_p + l * epipolar_direction;
        if (!isInliers_(px_cur)) {
            continue;
        }

        float ncc = ComputeNCCScores_(ref_p, px_cur);
        if (ncc > best_ncc) {
            best_cur_p = px_cur;
            best_ncc = ncc;
        }
    }
    // cout << "best_ncc = " << best_ncc << endl;

    if (best_ncc < 0.85f) {
        return false;
    }

    cur_p = best_cur_p;
    return true;
}

bool EpipolarSearch::RunEpipolarSearch(const cv::Mat &ref_img, const cv::Mat &cur_img, 
    const cv::Mat &ref_p, cv::Mat &cur_p,
    const cv::Mat &R, cv::Mat &t) {

    mmRefImg_= ref_img.clone();
    mmCurImg_ = cur_img.clone();

    bool flag = RunEpipolarSearch(ref_p, cur_p, R, t);
    return flag;
}

void EpipolarSearch::SetupImage(const cv::Mat &ref_img, const cv::Mat &cur_img) {
    mmRefImg_ = ref_img.clone();
    mmCurImg_ = cur_img.clone();
}

void EpipolarSearch::PixelToCam_(const cv::Mat &pixel, cv::Mat &p3d) {
    p3d.at<float>(0) = (pixel.at<float>(0) - mmK_.at<float>(0,2)) / mmK_.at<float>(0,0);
    p3d.at<float>(1) = (pixel.at<float>(1) - mmK_.at<float>(1,2)) / mmK_.at<float>(1,1);
    p3d.at<float>(2) = 1;
}

void EpipolarSearch::CamToPixel_(const cv::Mat &p3d, cv::Mat &pixel) {
    pixel.at<float>(0) = (p3d.at<float>(0) / p3d.at<float>(2)) * mmK_.at<float>(0,0) + mmK_.at<float>(0,2);
    pixel.at<float>(1) = (p3d.at<float>(1) / p3d.at<float>(2)) * mmK_.at<float>(1,1) + mmK_.at<float>(1,2);
} 

bool EpipolarSearch::isInliers_(const cv::Mat &pixel) {
    if (pixel.at<float>(0) < 5 || pixel.at<float>(0) > mmCurImg_.cols - 5 ||
        pixel.at<float>(1) < 5 || pixel.at<float>(1) > mmCurImg_.rows - 5) {
        return false;
    }
    return true;
}

// compute NCC scores
float EpipolarSearch::ComputeNCCScores_(const cv::Mat &ref_p, const cv::Mat &cur_p) {
    
    int radius = mnWin_ / 2;
    int ref_u = ref_p.at<float>(0);
    int ref_v = ref_p.at<float>(1);
    int cur_u = cur_p.at<float>(0);
    int cur_v = cur_p.at<float>(1);

    float sum_up = 0;
    float sum_down = 0;
    float sum_down_1 = 0, sum_down_2 = 0;
    for (int r = -radius; r < radius; ++r) {
        for (int c = -radius; c < radius; ++c) {
            float ref_img = mmRefImg_.at<float>(ref_v+r, ref_u+c);
            float cur_img = mmCurImg_.at<float>(cur_v+r, cur_u+c);

            sum_up += ref_img * cur_img;
            sum_down_1 += ref_img * ref_img;
            sum_down_2 += cur_img * cur_img;
        }
    }
    sum_down = sqrt(sum_down_1 * sum_down_2);
    float ncc = sum_up / sum_down;

    return ncc;
}