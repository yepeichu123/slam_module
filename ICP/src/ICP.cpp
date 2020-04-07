#include "ICP.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <Eigen/Core>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

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
        cout << "SovleICPByBA." << endl;
        cout << "And we first need to estimate the pose by SVD." << endl;
        cv::Mat temp_R, temp_t;
        flag = SolveICPBySVD(ref_p3d, cur_p3d, temp_R, temp_t);
        
        // convert mat to eigen
        Eigen::Matrix3d Rot;
        Eigen::Vector3d trans;
        ConvertPoseMatToEigen_(temp_R, temp_t, Rot, trans);

        // BA optimization
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(true);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(
                g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType> >()
            )
        );
        optimizer.setAlgorithm(solver);

        // add pose vertex 
        g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
        g2o::SE3Quat pose(Rot, trans);
        v_se3->setId(0);
        v_se3->setEstimate(pose);
        optimizer.addVertex(v_se3);

        int count = 1;
        for (int i = 0; i < ref_p3d.size(); ++i) {
            Eigen::Vector3d p_to, p_from;
            ConvertPoint3fToEigen_(ref_p3d[i], p_to);
            ConvertPoint3fToEigen_(cur_p3d[i], p_from);

            // construct point vertex 
            g2o::VertexSBAPointXYZ* v_p3d = new g2o::VertexSBAPointXYZ();
            v_p3d->setId(count++);
            v_p3d->setEstimate(p_from);
            // v_p3d->setMarginalized(true);
            optimizer.addVertex(v_p3d);

            EdgeICP* e = new EdgeICP();
            e->setVertex(0, v_se3);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p3d));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(0)->second));
            e->setMeasurement(p_to);
            e->setInformation(Eigen::Matrix3d::Identity());
            optimizer.addEdge(e);
        }

        cout << "Begin to optimization!" << endl;
        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;
        optimizer.optimize(10);
        
        g2o::SE3Quat T = v_se3->estimate();
        Rot = T.rotation();
        trans = T.translation();
        ConvertPoseEigenToMat_(Rot, trans, R, t);
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
    }
}

void ICP::ConvertPoseMatToEigen_(const cv::Mat& R, const cv::Mat& t, Eigen::Matrix3d& Rot, Eigen::Vector3d& trans) {
    cv::Mat new_R, new_t;
    R.convertTo(new_R, CV_64F);
    t.convertTo(new_t, CV_64F);
    Rot << new_R.at<double>(0, 0), new_R.at<double>(0, 1), new_R.at<double>(0, 2),
           new_R.at<double>(1, 0), new_R.at<double>(1, 1), new_R.at<double>(1, 2),
           new_R.at<double>(2, 0), new_R.at<double>(2, 1), new_R.at<double>(2, 2);
    trans << new_t.at<double>(0), new_t.at<double>(1), new_t.at<double>(2);
}

void ICP::ConvertPoseEigenToMat_(const Eigen::Matrix3d& Rot, const Eigen::Vector3d& trans, cv::Mat& R, cv::Mat& t) {
    R = (cv::Mat_<float>(3, 3) << Rot(0, 0), Rot(0, 1), Rot(0, 2),
                                  Rot(1, 0), Rot(1, 1), Rot(1, 2),
                                  Rot(2, 0), Rot(2, 1), Rot(2, 2));
    t = (cv::Mat_<float>(3, 1) << trans.x(), trans.y(), trans.z());
}

void ICP::ConvertPoint3fToEigen_(const cv::Point3f& p3d, Eigen::Vector3d& point3d) {
    point3d << (double)p3d.x, (double)p3d.y, (double)p3d.z;
}