#include "PNP.h"
#include "pnp_g2o.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;

PNP::PNP(const cv::Mat &K) {
    m_K_ = K.clone();

    cv::Mat K_new;
    m_K_.convertTo(K_new, CV_64F);
    m_eK_ << K_new.at<double>(0, 0), K_new.at<double>(0,1), K_new.at<double>(0,2),
             K_new.at<double>(1, 0), K_new.at<double>(1,1), K_new.at<double>(1,2),
             K_new.at<double>(2, 0), K_new.at<double>(2,1), K_new.at<double>(2,2);
}

PNP::~PNP() {

}

bool PNP::RunPNP(std::vector<cv::Point3f> &ref_p3d, std::vector<cv::Point2f> &cur_p2d,
    cv::Mat &R, cv::Mat &t, const SolvePnpType& type) {
    
    bool flag = false;
    if (ref_p3d.size() > 0 && cur_p2d.size() > 0 && ref_p3d.size() == cur_p2d.size()) {
        
        // compute relative pose
        switch(type) {
            case SolvePnpType::PNP_CV: {
                cout << "We choos PNP_CV method to compute relative pose." << endl;
                flag = ComputeRelativePosePNP_(ref_p3d, cur_p2d, R, t);
                break;
            }
            case SolvePnpType::PNP_BA: {
                cout << "We choose PNP_BA method to compute relative pose." << endl;
                flag = ComputeRelativePosePNPByBA_(ref_p3d, cur_p2d, R, t);
                break;
            }
            case SolvePnpType::PNP_COMB: {
                cout << "We choose PNP_COMB method to compute relative pose." << endl;
                cout << "So we first use CV method to compute the initial relative pose, and then use BA to optimize." << endl;
                if (ComputeRelativePosePNP_(ref_p3d, cur_p2d, R, t)) {
                    flag = ComputeRelativePosePNPByBA_(ref_p3d, cur_p2d, R, t);
                }
                break;
            }
            default: {
                break;
            }
        }
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

    bool flag = false;
    if (ref_p3d.size() > 0 && cur_p2d.size() > 0 && ref_p3d.size() == cur_p2d.size()) {
        cout << "Enter ComputeRelativePosePNP_" << endl;
        vector<char> inliers;
        cv::Mat r_vec, t_vec;
        if (cv::solvePnPRansac(ref_p3d, cur_p2d, m_K_, cv::Mat(), r_vec, t_vec, false, 100, 1.0F, 0.99, inliers, cv::SOLVEPNP_EPNP)) {
            flag = true;
            cv::Mat r;
            cv::Rodrigues(r_vec, r);
            r.convertTo(R, CV_32F);
            t_vec.convertTo(t, CV_32F);
            float e = CheckPoseError(ref_p3d, cur_p2d, R, t);
            cout << "PNP_CV error is: " << e << endl;
        }        
    }
    return flag;
}

bool PNP::ComputeRelativePosePNPByBA_(std::vector<cv::Point3f>& ref_p3d, std::vector<cv::Point2f>& cur_p2d,
    cv::Mat& R, cv::Mat& t) {
    
    bool flag = false;

    if (ref_p3d.size() > 0 && cur_p2d.size() > 0 && ref_p3d.size() == cur_p2d.size()) {
        cout << "Enter ComputeRelativePosePNPByBA_" << endl;
        cv::Mat R_temp, t_temp;

        if (!R.empty() && !t.empty()) {
            R_temp = R.clone();
            t_temp = t.clone();
        }
        else {
            R_temp = cv::Mat::eye(3, 3, CV_64F);
            t_temp = cv::Mat::zeros(3, 1, CV_64F);
        }
        Eigen::Matrix3d Rot;
        Eigen::Vector3d trans;
        ConvertPoseMatToEigen_(R_temp, t_temp, Rot, trans);

        // setup algorithm
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(true);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(
                g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> >()
            )
        );
        optimizer.setAlgorithm(solver);

        // add vertex 
        g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
        v_se3->setId(0);
        g2o::SE3Quat pose(Rot, trans);
        v_se3->setEstimate(pose);
        optimizer.addVertex(v_se3);
        
        int count = 1;
        bool fixed_point = false;
        // add edges
        for (int i = 0; i < ref_p3d.size(); ++i) {
            Eigen::Vector3d p3d;
            Eigen::Vector2d p2d;
            ConvertPointMatToEigen_(ref_p3d[i], cur_p2d[i], p3d, p2d);

            g2o::VertexSBAPointXYZ* vp3d = new g2o::VertexSBAPointXYZ();
            vp3d->setId(count++);
            vp3d->setEstimate(p3d);
            if (fixed_point) {
                vp3d->setFixed(true);
            }
            else {
                vp3d->setMarginalized(true);
            }
            
            optimizer.addVertex(vp3d);

            EdgePNP2* epnp = new EdgePNP2(m_eK_);
            epnp->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vp3d));
            epnp->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(0)->second));
            epnp->setMeasurement(p2d);
            epnp->setInformation(Eigen::Matrix2d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            rk->setDelta(1.0);
            epnp->setRobustKernel(rk);

            optimizer.addEdge(epnp);
        } 
        cout << "Begin optimization." << endl;
        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;
        optimizer.optimize(100);

        g2o::SE3Quat T = v_se3->estimate();
        Rot = T.rotation();
        trans = T.translation();
        ConvertPoseEigenToMat_(Rot, trans, R, t);

        // update points
        vector<cv::Point3f> new_ref_p3d;
        if (!fixed_point) {
            for (int i = 1; i < count; ++i) {
                g2o::VertexSBAPointXYZ* vp3d = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertices().find(i)->second);
                Eigen::Vector3d point = vp3d->estimate();
                cv::Point3f ref_p(point(0), point(1), point(2));
                new_ref_p3d.push_back(ref_p);
            }
            if (new_ref_p3d.size() > 0) {
                ref_p3d.clear();
                ref_p3d.insert(ref_p3d.end(), new_ref_p3d.begin(), new_ref_p3d.end());
                cout << "Update 3d points, we have " << ref_p3d.size() << " matching pairs left." << endl;
            }
        }
        // Todo: Check inliers
        // and then return the state
        flag = true;
        float e = CheckPoseError(ref_p3d, cur_p2d, R, t);
        cout << "PNP_BA error is: " << e << endl;
    }
    return flag;
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

void PNP::ConvertPoseMatToEigen_(const cv::Mat& R, const cv::Mat &t, Eigen::Matrix3d& Rot, Eigen::Vector3d& trans) {
    cv::Mat R_new, t_new;
    R.convertTo(R_new, CV_64F);
    t.convertTo(t_new, CV_64F);

    Rot << R_new.at<double>(0,0), R_new.at<double>(0,1), R_new.at<double>(0,2),
           R_new.at<double>(1,0), R_new.at<double>(1,1), R_new.at<double>(1,2),
           R_new.at<double>(2,0), R_new.at<double>(2,1), R_new.at<double>(2,2);

    trans << t_new.at<double>(0), t_new.at<double>(1), t_new.at<double>(2);
}

void PNP::ConvertPoseEigenToMat_(const Eigen::Matrix3d& Rot, const Eigen::Vector3d& trans, cv::Mat& R, cv::Mat& t) {
    R = (cv::Mat_<float>(3, 3) << Rot(0, 0), Rot(0, 1), Rot(0, 2),
                                  Rot(1, 0), Rot(1, 1), Rot(1, 2),
                                  Rot(2, 0), Rot(2, 1), Rot(2, 2));
    t = (cv::Mat_<float>(3, 1) << trans(0), trans(1), trans(2));
}

void PNP::ConvertPointMatToEigen_(const cv::Point3f& ref_p3d, const cv::Point2f& cur_p2d, Eigen::Vector3d& p3d, Eigen::Vector2d& p2d) {
    p3d << ref_p3d.x, ref_p3d.y, ref_p3d.z;
    p2d << cur_p2d.x, cur_p2d.y;
}