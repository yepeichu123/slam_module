#include "PNP.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace std;

PNP::PNP(const cv::Mat &K) {
    m_K_ = K.clone();
}

PNP::~PNP() {

}

bool PNP::RunPNP(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
    cv::Mat &R, cv::Mat &t) {
    
    if (ref_p3d.size() > 0 && cur_p2d.size() > 0 && ref_p3d.size() == cur_p2d.size()) {
        // compute relative pose
        bool flag = ComputeRelativePosePNP_(ref_p3d, cur_p2d, R, t);

        if (flag) {
            // compute error 
            float error = CheckPoseError(ref_p3d, cur_p2d, R, t);
        }

        return flag;
    }

    return false;
}

float PNP::CheckPoseError(const std::vector<cv::Point3f> &ref_p3d, const std::vector<cv::Point2f> &cur_p2d,
    const cv::Mat &R, const cv::Mat &t) {
    
    float e = 0;
    if (ref_p3d.size() == cur_p2d.size() && ref_p3d.size() > 0 && cur_p2d.size() > 0) {
        for (int i = 0; i < ref_p3d.size(); ++i) {
            cv::Mat p3d = (cv::Mat_<float>(3,1) << ref_p3d[i].x, ref_p3d[i].y, ref_p3d[i].z);
            cv::Mat p2d = (cv::Mat_<float>(2,1) << cur_p2d[i].x, cur_p2d[i].y);

            cv::Mat proj_pixel;
            Cam2Pixel_(R*p3d + t, proj_pixel);
            cv::Mat error = p2d - proj_pixel;
            // cout << i << " : " << error << endl;
            e += cv::norm(error);
        }
        e /= ref_p3d.size();
        cout << "PNP average error is : " << e << endl;
        return e; 
    }
    e = 1.0e+5;
    cout << "Oh man, please check your input data, maybe there is something wrong." << endl;
    return e;
}

bool PNP::ComputeRelativePosePNP_(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
    cv::Mat &R, cv::Mat &t) {

    if (ref_p3d.size() > 0 && cur_p2d.size() > 0 && ref_p3d.size() == cur_p2d.size()) {
        vector<char> inliers;
        cv::Mat r_vec, t_vec;
        if (cv::solvePnPRansac(ref_p3d, cur_p2d, m_K_, cv::Mat(), r_vec, t_vec, false, 100, 8.0F, 0.99, inliers)) {
            ChooseGoodMatching(ref_p3d, cur_p2d, inliers);
            cv::Mat r;
            cv::Rodrigues(r_vec, r);
            r.convertTo(R, CV_32F);
            t_vec.convertTo(t, CV_32F);
            cout << "R = " << R << "\n t = " << t << endl;
            return true;
        }        
    }

    return false;
}

void PNP::ChooseGoodMatching(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
    std::vector<char> &inliers) {
     if (ref_p3d.size() == cur_p2d.size() && ref_p3d.size() > 0 && cur_p2d.size() > 0 && inliers.size() > 0) {
         vector<cv::Point3f> new_ref_p3d;
         vector<cv::Point2f> new_cur_p2d;

         for (int i = 0; i < inliers.size(); ++i) {
            if (inliers[i] > 0) {
                new_ref_p3d.push_back(ref_p3d[i]);
                new_cur_p2d.push_back(cur_p2d[i]);
            }
         }

         if (new_ref_p3d.size() > 0 && new_cur_p2d.size() > 0) {
            ref_p3d.clear();
            cur_p2d.clear();
            ref_p3d.insert(ref_p3d.end(), new_ref_p3d.begin(), new_ref_p3d.end());
            cur_p2d.insert(cur_p2d.end(), new_cur_p2d.begin(), new_cur_p2d.end());
         }
     }   
}

void PNP::Cam2Pixel_(const cv::Mat &p3d, cv::Mat &pixel) {
    float z = p3d.at<float>(2);
    pixel = (cv::Mat_<float>(2, 1) << 
        (p3d.at<float>(0) / z * m_K_.at<float>(0,0) + m_K_.at<float>(0,2)),
        (p3d.at<float>(1) / z * m_K_.at<float>(1,1) + m_K_.at<float>(1,2))
    );
}

void PNP::Pixel2Cam_(const cv::Mat &pixel, cv::Mat &p3d) {
    p3d = (cv::Mat_<float>(3, 1) << 
        (pixel.at<float>(0) - m_K_.at<float>(0,2)) / m_K_.at<float>(0,0),
        (pixel.at<float>(1) - m_K_.at<float>(1,2)) / m_K_.at<float>(1,1),
        1    
    );
}