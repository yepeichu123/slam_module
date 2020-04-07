#include "ICP_G2O.h"
using namespace std;
using namespace g2o;

VertexPose::VertexPose() {

}

void VertexPose::setToOriginImpl() {
    _estimate = g2o::SE3Quat();
}

/// left multiplication on SE3
void VertexPose::oplusImpl(const double *update_) {
    Eigen::Map<const Eigen::Matrix<double, 6, 1> > update(update_);
    _estimate = g2o::SE3Quat::exp(update) * _estimate;
}

bool VertexPose::read(std::istream &in) {
    return false;
}

bool VertexPose::write(std::ostream &out) const {
    return false;
}



EdgeICP::EdgeICP() {
}

bool EdgeICP::read(std::istream& is) {
    cerr << "We have not impletemented read function." << endl;
    return false;
}

bool EdgeICP::write(std::ostream& os) const {
    cerr << "We have not impletemented write function." << endl;
    return false;
}

void EdgeICP::computeError() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Eigen::Vector3d obs(_measurement);
    _error = obs - pose->estimate().map(point->estimate());
}

void EdgeICP::linearizeOplus() {
    const VertexSE3Expmap* pose = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* point = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate());
    Eigen::Vector3d xyz = point->estimate();
    Eigen::Vector3d xyz_trans = T.map(xyz);
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    // jacobian of points
    _jacobianOplusXi = -T.rotation().toRotationMatrix();

    // jacobian of pose (Tp)^
    _jacobianOplusXj(0, 0) = -1;
    _jacobianOplusXj(0, 1) = 0;
    _jacobianOplusXj(0, 2) = 0;
    _jacobianOplusXj(0, 3) = 0;
    _jacobianOplusXj(0, 4) = -z; 
    _jacobianOplusXj(0, 5) = y;

    _jacobianOplusXj(1, 0) = 0;
    _jacobianOplusXj(1, 1) = -1;
    _jacobianOplusXj(1, 2) = 0;
    _jacobianOplusXj(1, 3) = z;
    _jacobianOplusXj(1, 4) = 0;
    _jacobianOplusXj(1, 5) = -x;

    _jacobianOplusXj(2, 0) = 0;
    _jacobianOplusXj(2, 1) = 0;
    _jacobianOplusXj(2, 2) = -1;
    _jacobianOplusXj(2, 3) = -y;
    _jacobianOplusXj(2, 4) = x;
    _jacobianOplusXj(2, 5) = 0;
}