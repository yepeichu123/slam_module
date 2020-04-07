#ifndef ICP_G2O_H
#define ICP_G2O_H

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>
#include <iostream>

class VertexPose: public g2o::BaseVertex<6, g2o::SE3Quat> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPose();

        virtual void setToOriginImpl();

        /// left multiplication on SE3
        virtual void oplusImpl(const double *update_);

        virtual bool read(std::istream &in);

        virtual bool write(std::ostream &out) const;
};


class EdgeICP: public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap> { 
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeICP();

        bool read(std::istream& is);

        bool write(std::ostream& os) const;

        virtual void computeError();

        virtual void linearizeOplus();
    
    private:
        // Eigen::Vector3d m_p3d_;
};

#endif // ICP_G2O_H