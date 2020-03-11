#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class FeatureExtraction {
    public:
        // default constructor
        FeatureExtraction(const std::string &feat_types, const cv::Mat &img);

        // default deconstructor
        ~FeatureExtraction();

        // set up image
        void SetImage(const cv::Mat &img);

        // divide image into many parts
        void setDivedeImg_(const int &rows, const int &cols);

        // run feature extraction with divide image
        // actually, I didn't divide image very well, and I used OpenCV libs to extract features from dividing images.
        void runFeatureExtractDiv(std::vector<cv::KeyPoint> &kpts, cv::Mat &desp);

        // run feature extraction without divide image, just using OpenCV libs to extract original image.
        void runFeatureExtractNoDiv(std::vector<cv::KeyPoint> &kpts, cv::Mat &desp);
    private:

        // divide image
        void divideImg_(cv::Mat &in_img, std::vector<cv::Mat> &out_img);

        // feature detection
        void featureDetect_(std::vector<cv::Mat> &sub_imgs, std::vector<cv::KeyPoint> &kpts);

        // rows and cols for dividing the image
        int mnRows_;
        int mnCols_;
        int mnRowSize_;
        int mnColSize_;

        // feature detecting and extracting pointer
        cv::Ptr<cv::FeatureDetector> mpDetector_;
        cv::Ptr<cv::DescriptorExtractor> mpDescriptor_;

        // input image
        cv::Mat mmImg_;
};


#endif // FEATURE_EXTRACTION_H