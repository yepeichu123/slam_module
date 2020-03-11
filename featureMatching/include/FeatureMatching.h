#ifndef FEATURE_MATCHING_H
#define FEATURE_MATCHING_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class FeatureMatching {
    public:
        // default constructor
        FeatureMatching(const std::string &matching_types, const std::string &feat_types);

        // defalut deconstructor
        ~FeatureMatching();

        // run feature matching 
        void RunFeatureMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches);

        // run feature matching by brute force matching
        void RunBFMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches);

        // run feature matching by flann matching 
        void RunFlannMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches);

    private:
        // Flann matching pointer
        cv::Ptr<cv::FlannBasedMatcher> mpFlannMatcher_;

        // Flann for orb
        cv::Ptr<cv::flann::Index> mpFlannIndex_;

        // Brute force matching pointer
        cv::Ptr<cv::BFMatcher> mpBFMatcher_;

        // matching type
        std::string msMatchingTypes_;

        // feature type 
        std::string msFeatureTypes_;
};

#endif // FEATURE_MATCHING_H