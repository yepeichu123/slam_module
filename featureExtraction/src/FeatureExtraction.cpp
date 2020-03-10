#include "FeatureExtraction.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;

FeatureExtraction::FeatureExtraction(const std::string &feat_types, const cv::Mat &img) {
    if (img.empty()) {
        cout << "Input empty image, please check again." << endl;
        return;
    }
    mmImg_ = img.clone();

    if (feat_types == "ORB") {
        cout << "Choose ORB features!" << endl;
        mpDetector_ = cv::ORB::create(10000);
        mpDescriptor_ = cv::ORB::create();
    }
    else if (feat_types == "SIFT") {
        cout << "Choose SIFT features!" << endl;
        mpDetector_ = cv::xfeatures2d::SIFT::create(10000);
        mpDescriptor_ = cv::xfeatures2d::SIFT::create();
    }
    else if (feat_types == "SURF") {
        cout << "Choose SURF features!" << endl;
        mpDetector_ = cv::xfeatures2d::SURF::create(10000);
        mpDescriptor_ = cv::xfeatures2d::SURF::create();
    }
    else if (feat_types == "AKAZE") {
        cout << "Choose AKAZE features!" << endl;
        mpDetector_ = cv::AKAZE::create();
        mpDescriptor_ = cv::AKAZE::create();
    }
    else if (feat_types == "BRIEF") {
        cout << "Choos BRIEF features!" << endl;
        mpDetector_ = cv::ORB::create(10000);
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extract = cv::xfeatures2d::BriefDescriptorExtractor::create();
        mpDescriptor_ = extract;
    }
    else {
        cout << "Please input one of following feature types : ORB, SIFT, SURF, AKAZE, BRIEF." << endl;
        return;
    }

    mnRows_ = 2;
    mnCols_ = 3;
}

FeatureExtraction::~FeatureExtraction() {

}

void FeatureExtraction::setDivedeImg_(const int &rows, const int &cols) {
    mnRows_ = rows;
    mnCols_ = cols;
}

void FeatureExtraction::runFeatureExtractDiv(std::vector<cv::KeyPoint> &kpts, cv::Mat &desp) {
    cout << "Enter FeatureExtraction modules with dividing image." << endl;
    if (mmImg_.empty()) {
        cout << "empty image! return!" << endl;
        return;
    }

    std::vector<cv::Mat> subImgs;
    divideImg_(mmImg_, subImgs);
    cout << "We get " << subImgs.size() << " sub images from input image!" << endl;

    featureDetect_(subImgs, kpts);

    // mpDetector_->detect(mmImg_, kpts);
    mpDescriptor_->compute(mmImg_, kpts, desp);
    cout << "Detect " << kpts.size() << " keypoints in image!" << endl;

}

void FeatureExtraction::runFeatureExtractNoDiv(std::vector<cv::KeyPoint> &kpts, cv::Mat &desp) {
    cout << "Enter FeatureExtraction modules without dividing image." << endl;
    if (mmImg_.empty()) {
        cout << "empty image! return!" << endl;
        return;
    }

    mpDetector_->detect(mmImg_, kpts);
    mpDescriptor_->compute(mmImg_, kpts, desp);
    cout << "Detect " << kpts.size() << " keypoints in image!" << endl;
}

void FeatureExtraction::divideImg_(cv::Mat &in_img, std::vector<cv::Mat> &out_img) {
    cout << "Enter divide Image program!" << endl;

    if (in_img.empty()) {
        cout << "empty image! error!" << endl;
        return;
    }

    int row = in_img.rows;
    int col = in_img.cols;
    mnRowSize_ = ceil((double)row / mnRows_);
    mnColSize_ = ceil((double)col / mnCols_);
    int r_size_cp = mnRowSize_;
    int c_size_cp = mnColSize_;
    cout << "rowsize = " << mnRowSize_ << ", colsize = " << mnColSize_ << endl;
    cout << "Row = " << mnRows_ << ", Col = " << mnCols_ << endl;

    cv::Mat new_img;
    int padding = 4;
    new_img.create(row + padding*2, col + padding*2, CV_8UC1);
    cv::copyMakeBorder(in_img, new_img, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));

    for (int r = 0; r < mnRows_; ++r) {
        for (int c = 0; c < mnCols_; ++c) {
            /*
            if ((c+1)*mnColSize_ > col) {
                mnColSize_ = col - c*mnColSize_;
            }
            if ((r+1)*mnRowSize_ > row) {
                mnRowSize_ = row - r*mnRowSize_;
            }*/
            cv::Mat temp_img;
            temp_img.create(mnRowSize_, mnColSize_, CV_8UC1);
            cv::Rect rect(c*mnColSize_, r*mnRowSize_, mnColSize_, mnRowSize_);
            new_img(rect).copyTo(temp_img);
            cv::imshow("sub_img", temp_img);
            cv::waitKey(1);
            out_img.push_back(temp_img);

            mnRowSize_ = r_size_cp;
            mnColSize_ = c_size_cp;
        }
    }
    cout << "Finally, we get " << out_img.size() << " sub images!" << endl;
}

void FeatureExtraction::featureDetect_(std::vector<cv::Mat> &sub_imgs, std::vector<cv::KeyPoint> &kpts) {
    cout << "Enter feature detection function." << endl;
    for (int r = 0; r < mnRows_; ++r) {
        for (int c = 0; c < mnCols_; ++c) {
            // cout << "Enter feature detection : " << r*mnCols_ + c << endl;
            cv::Mat temp_img = sub_imgs[r*mnCols_ + c].clone();

            std::vector<cv::KeyPoint> sub_kpts;
            mpDetector_->detect(temp_img, sub_kpts);
            // cout << "detect " << sub_kpts.size() << " points!" << endl;
            for (int i = 0; i < sub_kpts.size(); ++i) {
                sub_kpts[i].pt.x += c * mnColSize_;
                sub_kpts[i].pt.y += r * mnRowSize_;
            }
            kpts.insert(kpts.end(), sub_kpts.begin(), sub_kpts.end());
        }
    }
    cout << "Finally, we get " << kpts.size() << " keypoints from sub images!" << endl;
}