#include "EpipolarConstrain.h"

EpipolarConstrain::EpipolarConstrain(const cv::Mat &K) {
    mK_ = K.clone();
}

EpipolarConstrain::~EpipolarConstrain() {

}

bool EpipolarConstrain::ComputeRelativePose() {

}

int EpipolarConstrain::MatchingByRANSAC() {

}

float EpipolarConstrain::CheckPoseError() {

}