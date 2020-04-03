#include "ICP.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

ICP::ICP() {

}

ICP::~ICP() {
    
}

bool ICP::RunICP(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
    cv::Mat& R, cv::Mat &t, const ICP_TYPE& type) {

    bool flag = false;
    if (ref_p3d.size() > 0 && cur_p3d.size() > 0 && ref_p3d.size() == cur_p3d.size()) {
        switch(type) {
            case ICP_TYPE::BA: {
                flag = SovleICPByBA(ref_p3d, cur_p3d, R, t);
                break;
            }
            case ICP_TYPE::SVD: {
                flag = SolveICPBySVD(ref_p3d, cur_p3d, R, t);
                break;
            }
            default: {
                break;
            }
        }

        if (flag) {
            float e = CheckPoseError(ref_p3d, cur_p3d, R, t);
            std::cout << "Average relative pose error is : " << e << std::endl;
        }
    }
    return flag;
}

bool ICP::SolveICPBySVD(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
    cv::Mat& R, cv::Mat &t) {
    
    bool flag = false;
    if (ref_p3d.size() > 0 && cur_p3d.size() > 0 && ref_p3d.size() == cur_p3d.size()) {
        std::cout << "SolveICPBySVD." << std::endl;

        std::vector<cv::Mat> ref_pm, cur_pm;
        ConvertP3dToMatVec_(ref_p3d, ref_pm);
        ConvertP3dToMatVec_(cur_p3d, cur_pm);

        // compute the point centroid
        std::cout << "Compute point centroid." << std::endl;
        cv::Mat ref_cen = cv::Mat::zeros(3, 1, CV_32F);
        cv::Mat cur_cen = cv::Mat::zeros(3, 1, CV_32F);
        for (int i = 0; i < ref_pm.size(); ++i) {
            ref_cen = ref_cen + ref_pm[i];
            cur_cen = cur_cen + cur_pm[i];
        }
        ref_cen = ref_cen / ref_pm.size();
        cur_cen = cur_cen / ref_pm.size();

        // compute bias coordinate
        std::cout << "Compute bias coordinate." << std::endl;
        std::vector<cv::Mat> ref_bias, cur_bias;
        for (int i = 0; i < ref_pm.size(); ++i) {
            ref_bias.push_back(ref_pm[i] - ref_cen);
            cur_bias.push_back(cur_pm[i] - cur_cen);
        }

        // compute rotation
        std::cout << "Compute Rotation." << std::endl;
        cv::Mat W = cv::Mat::zeros(3, 3, CV_32F);
        for (int i = 0; i < ref_bias.size(); ++i) {
            W = W + ref_bias[i] * cur_bias[i].t();
        }
        cv::Mat w, u, vt;
        cv::SVD::compute(W, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
        cv::Mat r_vec = u * vt;

        // compute translation
        std::cout << "Compute translation." << std::endl;
        cv::Mat t_vec = ref_cen - r_vec * cur_cen;

        // convert type 
        r_vec.convertTo(R, CV_32F);
        t_vec.convertTo(t, CV_32F);

        flag = true;
    }
    return flag;
}

bool ICP::SovleICPByBA(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
    cv::Mat& R, cv::Mat &t) {

    // Todo:
    // Compute initialized value by SVD 
    // and then simultaneously optimize points and pose 
    bool flag = false;
    
    if (ref_p3d.size() > 0 && cur_p3d.size() > 0 && ref_p3d.size() == cur_p3d.size()) {
        cv::Mat R, t;
        flag = SolveICPBySVD(ref_p3d, cur_p3d, R, t);

        // BA optimization
        // ...
    }
    

    return flag;
}

float ICP::CheckPoseError(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point3f>& cur_p3d,
    cv::Mat& R, cv::Mat &t) {
    float error = 0;

    if (!R.empty() && !t.empty()) {
        std::vector<cv::Mat> ref_pm, cur_pm;
        ConvertP3dToMatVec_(ref_p3d, ref_pm);
        ConvertP3dToMatVec_(cur_p3d, cur_pm);

        for (int i = 0; i < ref_pm.size(); ++i) {
            cv::Mat e = ref_pm[i] - (R * cur_pm[i] + t);
            error += cv::norm(e);
        }
        error /= ref_pm.size();
    }

    return error;
}

void ICP::ConvertP3dToMat_(const cv::Point3f& p3d, cv::Mat& mp3d) {
    mp3d = (cv::Mat_<float>(3, 1) << p3d.x, p3d.y, p3d.z);
}

void ICP::ConvertP3dToMatVec_(const std::vector<cv::Point3f>& vp3d, std::vector<cv::Mat>& vmp3d) {
    if (vp3d.size() > 0) {
        vmp3d.clear();
        for (int i = 0; i < vp3d.size(); ++i) {
            cv::Mat p = (cv::Mat_<float>(3,1) << vp3d[i].x, vp3d[i].y, vp3d[i].z);
            vmp3d.push_back(p);
        }
        std::cout << "We convert " << vmp3d.size() << " points from point3d to mat." << std::endl;
    }
}