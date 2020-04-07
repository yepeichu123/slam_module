#ifndef PNP_G2O_H
#define PNP_G2O_H

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>

class EdgePNP: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgePNP(const Eigen::Vector3d& p3d, const Eigen::Matrix3d& K);

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;

        virtual void computeError();

        virtual void linearizeOplus();

    protected:
        Eigen::Vector3d m_p3d_;

        Eigen::Matrix3d m_K_;
};

class EdgePNP2: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgePNP2(const Eigen::Matrix3d& K);

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;

        virtual void computeError();

        virtual void linearizeOplus();

    protected:
        Eigen::Vector3d m_p3d_;

        Eigen::Matrix3d m_K_;
};

#endif //PNP_G2O_H