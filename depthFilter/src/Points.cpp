#include "Points.h"

Points::Points(const int &id) {
    mnId_ = id;
    mDepth_ = 0;
    mDepthUncertain_ = 0;
}

Points::Points(const int &id, const cv::Mat &K, const cv::KeyPoint &kpts, const cv::Mat &desp, const float &depth) {
    mnId_ = id;
    mK_ = K.clone();
    mKpt_ = kpts;
    mDesp_ = desp.clone();
    mDepth_ = depth;
    mDepthUncertain_ = 0;
}

Points::~Points() {

}

void Points::SetPointDepth(const float &depth) {
    mDepth_ = depth;
}

void Points::SetKeyPoint(const cv::KeyPoint &kpts) {
    mKpt_ = kpts;
}

void Points::SetDescriptor(const cv::Mat &desp) {
    mDesp_ = desp.clone();
}

void Points::SetCameraIntrinsics(const cv::Mat &K) {
    mK_ = K.clone();
}

void Points::SetDepthUncertainty(const float &depth_un) {
    mDepthUncertain_ = depth_un;
}

void Points::UpdatePointDepth(const float &new_depth, const float &new_uncertainty) {
    mDepth_ = new_depth;
    mDepthUncertain_ = new_uncertainty;
}

int Points::GetPointId() {
    return mnId_;
}

cv::KeyPoint& Points::GetKeyPoint() {
    return mKpt_;
}

cv::Mat& Points::GetDescriptor() {
    return mDesp_;
}

float& Points::GetDepth() {
    return mDepth_;
}

float Points::GetUncertainty() {
    return mDepthUncertain_;
}