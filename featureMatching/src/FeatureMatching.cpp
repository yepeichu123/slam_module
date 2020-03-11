#include "FeatureMatching.h"
#include <iostream>
#include <algorithm>

FeatureMatching::FeatureMatching(const std::string &matching_types, const std::string &feat_types):
    msMatchingTypes_(matching_types), msFeatureTypes_(feat_types) {
    
    if (msMatchingTypes_ == "FLANN") {
        std::cout << "Using FLANN feature matching!" << std::endl;
        mpFlannMatcher_ = new cv::FlannBasedMatcher();
    } 
    else {
        std::cout << "Using Brute force feature matching!" << std::endl;
        if (msFeatureTypes_ == "ORB") {
            mpBFMatcher_ = new cv::BFMatcher(cv::NORM_HAMMING);
        }
        else {
            mpBFMatcher_ = new cv::BFMatcher();
        }
    }
}

FeatureMatching::~FeatureMatching() {
    
}

void FeatureMatching::RunFeatureMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches) {
    if (msMatchingTypes_ == "FLANN") {
        RunFlannMatching(desp_1, desp_2, matches);
    }
    else {
        RunBFMatching(desp_1, desp_2, matches);
    }
}

void FeatureMatching::RunBFMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches) {
    std::cout << "Using Brute force matching." << std::endl;
    std::vector<cv::DMatch> temp_match;
    mpBFMatcher_->match(desp_1, desp_2, temp_match);

    double min_dist = std::min_element(temp_match.begin(), temp_match.end(), 
        [](cv::DMatch &m1, cv::DMatch &m2) {
            return m1.distance < m2.distance;
        } )->distance;
    /*
    double max_dist = std::max_element(temp_match.begin(), temp_match.end(),
        [](cv::DMatch &m1, cv::DMatch &m2) {
            return m1.distance < m2.distance;
        })->distance;
    */
    for (int i = 0; i < temp_match.size(); ++i) {
        if (temp_match[i].distance < std::max(30.0, min_dist*2)) {
            matches.push_back(temp_match[i]);
        }
    }
    std::cout << "Using Brute force matching and we get " << matches.size() << " points!" << std::endl;
}

void FeatureMatching::RunFlannMatching(cv::Mat &desp_1, cv::Mat &desp_2, std::vector<cv::DMatch> &matches) {
    
    if (msFeatureTypes_ == "ORB") {
        std::cout << "Using Flann Matching using ORB." << std::endl;
        mpFlannIndex_ = new cv::flann::Index(desp_1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
        cv::Mat matchIndex(desp_2.rows, 2, CV_32SC1);
        cv::Mat matchDistance(desp_2.rows, 2, CV_32FC1);
        mpFlannIndex_->knnSearch(desp_2, matchIndex, matchDistance, 2, cv::flann::SearchParams());

        for (int i = 0; i < matchDistance.rows; ++i) {
            if (matchDistance.at<float>(i, 0) < 0.6 * matchDistance.at<float>(i, 1)) {
                cv::DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
                matches.push_back(dmatches);
            }
        }
    }
    else {
        std::cout << "Using Flann Matching using other features." << std::endl;
        std::vector<std::vector<cv::DMatch> > matchPoints;
        std::vector<cv::Mat> train_dsc(1, desp_1);
        mpFlannMatcher_->add(train_dsc);
        mpFlannMatcher_->train();
        mpFlannMatcher_->knnMatch(desp_2, matchPoints, 2);
        std::cout << "Total matching points : " << matchPoints.size() << std::endl;

        for (int i = 0; i < matchPoints.size(); ++i) {
            if (matchPoints[i][0].distance < 0.7 * matchPoints[i][1].distance) {
                matches.push_back(matchPoints[i][0]);
            }
        }
        // mpFlannMatcher_->match(desp_1, desp_2, matches, cv::Mat());
    }    
    std::cout << "Final matching points : " << matches.size() << std::endl;

}