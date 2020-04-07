#include "pnp_g2o.h"
#include <iostream>

using namespace g2o;
using namespace std;

EdgePNP::EdgePNP(const Eigen::Vector3d& p3d, const Eigen::Matrix3d& K) {
    m_p3d_ = p3d;
    m_K_ = K;
}

bool EdgePNP::read(std::istream& is) {
    std::cout << "We have not impletemented read function yet." << std::endl;
    return false;
}

bool EdgePNP::write(std::ostream& os) const {
    std::cout << "We have not impletemented write function yet." << std::endl;
    return false;
}

void EdgePNP::computeError() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate());
    Eigen::Vector3d xyz_proj = m_K_ * T.map(m_p3d_);
    xyz_proj /= xyz_proj(2);
    _error = _measurement - xyz_proj.head<2>();
    cout << "_error = " << _error << endl;
}

void EdgePNP::linearizeOplus() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate());
    Eigen::Vector3d xyz_trans = T.map(m_p3d_);
    double fx = m_K_(0, 0);
    double fy = m_K_(1, 1);
    double cx = m_K_(0, 2);
    double cy = m_K_(1, 2);
    double x = xyz_trans(0);
    double y = xyz_trans(1);
    double z = xyz_trans(2);
    double z2 = z*z;

    _jacobianOplusXi(0, 0) = -fx / z;
    _jacobianOplusXi(0, 1) = 0;
    _jacobianOplusXi(0, 2) = fx * x / z2;
    _jacobianOplusXi(0, 3) = fx * x * y / z2;
    _jacobianOplusXi(0, 4) = -fx - fx * x * x / z2;
    _jacobianOplusXi(0, 5) = fx * y / z;

    _jacobianOplusXi(1, 0) = 0;
    _jacobianOplusXi(1, 1) = -fy / z;
    _jacobianOplusXi(1, 2) = fy * y / z2;
    _jacobianOplusXi(1, 3) = fy + fy * y * y / z2;
    _jacobianOplusXi(1, 4) = -fy * x * y / z2;
    _jacobianOplusXi(1, 5) = -fy * x / z;

    /*
    double z2 = -1. / (z*z);

    _jacobianOplusXi(0, 0) = z2 * (z * m_K_(0, 0));
    _jacobianOplusXi(0, 1) = 0;
    _jacobianOplusXi(0, 2) = z2 * (-x * m_K_(0, 0));
    _jacobianOplusXi(0, 3) = z2 * (-x * y * m_K_(0, 0));
    _jacobianOplusXi(0, 4) = z2 * (z*z + x*x) * m_K_(0, 0);
    _jacobianOplusXi(0, 5) = z2 * (-y * z * m_K_(0, 0));

    _jacobianOplusXi(1, 0) = 0;
    _jacobianOplusXi(1, 1) = z2 * (z * m_K_(1, 1));
    _jacobianOplusXi(1, 2) = z2 * (-y * m_K_(1, 1));
    _jacobianOplusXi(1, 3) = z2 * (-z*z - y*y) * m_K_(1, 1);
    _jacobianOplusXi(1, 4) = z2 * (x * y * m_K_(1, 1));
    _jacobianOplusXi(1, 5) = z2 * (x * z * m_K_(1, 1));
    */
}



EdgePNP2::EdgePNP2(const Eigen::Matrix3d& K) {
    m_K_ = K;
}

bool EdgePNP2::read(std::istream& is) {
    return false;
}

bool EdgePNP2::write(std::ostream& os) const {
    return false;
}

void EdgePNP2::computeError() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate());
    Eigen::Vector3d p3d(point->estimate());
    Eigen::Vector3d xyz_proj = m_K_ * T.map(p3d);
    xyz_proj /= xyz_proj(2);
    _error = _measurement - xyz_proj.head<2>();
    cout << "_error = " << _error.transpose() << endl;
}

void EdgePNP2::linearizeOplus() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate());
    Eigen::Vector3d p3d(point->estimate());
    Eigen::Vector3d xyz_trans = T.map(p3d);
    double fx = m_K_(0, 0);
    double fy = m_K_(1, 1);
    double cx = m_K_(0, 2);
    double cy = m_K_(1, 2);
    double x = xyz_trans(0);
    double y = xyz_trans(1);
    double z = xyz_trans(2);
    double z2 = z*z;

    Eigen::Matrix3d Rot = T.rotation().toRotationMatrix();
    Eigen::Matrix<double, 2, 3> J_e_pc;
    J_e_pc(0, 0) = fx / z;
    J_e_pc(0, 1) = 0;
    J_e_pc(0, 2) = -fx * x / z2;
    J_e_pc(1, 0) = 0;
    J_e_pc(1, 1) = fy / z;
    J_e_pc(1, 2) = -fy * y / z2;
    _jacobianOplusXi = -J_e_pc * Rot;

    _jacobianOplusXj(0, 0) = -fx / z;
    _jacobianOplusXj(0, 1) = 0;
    _jacobianOplusXj(0, 2) = fx * x / z2;
    _jacobianOplusXj(0, 3) = fx * x * y / z2;
    _jacobianOplusXj(0, 4) = -fx - fx * x * x / z2;
    _jacobianOplusXj(0, 5) = fx * y / z;

    _jacobianOplusXj(1, 0) = 0;
    _jacobianOplusXj(1, 1) = -fy / z;
    _jacobianOplusXj(1, 2) = fy * y / z2;
    _jacobianOplusXj(1, 3) = fy + fy * y * y / z2;
    _jacobianOplusXj(1, 4) = -fy * x * y / z2;
    _jacobianOplusXj(1, 5) = -fy * x / z;
}