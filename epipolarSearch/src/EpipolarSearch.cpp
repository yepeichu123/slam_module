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
    float min_idepth = 0,  max_idepth = 0.01;
    cv::Mat new_R, new_t;
    R.convertTo(new_R, CV_32F);
    t.convertTo(new_t, CV_32F);
    cv::Mat pt = mmK_ * new_R * ref_p3d;
    cv::Mat min_cur_p = pt + min_idepth * mmK_ * new_t;
    cv::Mat max_cur_p = pt + max_idepth * mmK_ * new_t;
    
    float u_min = min_cur_p.at<float>(0) / min_cur_p.at<float>(2);
    float v_min = min_cur_p.at<float>(1) / min_cur_p.at<float>(2);
    float u_max = max_cur_p.at<float>(0) / max_cur_p.at<float>(2);
    float v_max = max_cur_p.at<float>(1) / max_cur_p.at<float>(2);

    // compute epipolar an direction   
    cv::Mat epipolar_line;
    epipolar_line = (cv::Mat_<float>(2,1) << u_max - u_min, v_max - v_min);
    float epipolar_length = sqrtf( (u_max-u_min)*(u_max-u_min) + (v_max-v_min)*(v_max- v_min) );
    float d = 1 / epipolar_length;
    float dx = u_max - u_min;
    float dy = v_max - v_min;

    cv::Mat best_cur_p;
    float best_ncc = 0;
    for (float l = 0; l <= epipolar_length; l+=0.1) {
        cv::Mat px_cur;
        px_cur = (cv::Mat_<float>(2,1) << u_min + l * d * dx, v_min + l * d * dy);
        // cout << "px_cur = " << px_cur << endl;
        if (!isInliers_(px_cur)) {
            continue;
        }

        float ncc = ComputeNCCScores_(ref_p, px_cur);
        if (ncc > best_ncc) {
            best_cur_p = px_cur;
            best_ncc = ncc;
        }
    }

    if (best_ncc < 0.95f) {
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

    if (ncc >= 0 && ncc <= 1) {
        return ncc;
    }
    else {
        return 0;
    }
}