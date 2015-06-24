#include "map_localize/EdgeTrackingUtil.h"
#include "TooN/TooN.h"
#include "TooN/SVD.h"       // for SVD
#include "TooN/so3.h"       // for special orthogonal group
#include "TooN/se3.h"       // for special Euclidean group
#include "TooN/wls.h"       // for weighted least square


using namespace TooN;

void EdgeTrackingUtil::getEstimatedPoseIRLS(Eigen::Matrix4f& pose_cur, const Eigen::Matrix4f& pose_pre, const std::vector<SamplePoint>& vSamplePt, const Eigen::Matrix3f& intrinsics)
{
  double maxd_ = 10.;
  double alpha_ = 64.;
  // use a numerical non-linear optimization (weighted least square) to find pose (P)

  Matrix<3,3, double> R;
  Vector<3> t;
  for(int r=0; r<3; r++)
  {
    t[r] = pose_pre(r,3);
    for(int c=0; c<3; c++)
    {
      R(r,c) = pose_pre(r,c);
    }
  }
  SE3<double> se3_prev(SO3<double>(R),t);

  WLS<6> wls;
  for(int i=0; i<int(vSamplePt.size()); i++)
  {
    if(vSamplePt[i].dist < maxd_)
    {
      // INVERSE 1/(alpha_ + dist)
      wls.add_mJ(
        vSamplePt[i].dist,
        calcJacobian(vSamplePt[i].coord3, vSamplePt[i].coord2, vSamplePt[i].nuv, vSamplePt[i].dist, se3_prev, intrinsics),
        1.0/(static_cast<double>(alpha_) + (vSamplePt[i].dist>0 ? vSamplePt[i].dist : -vSamplePt[i].dist))
      );
    }
  }

  wls.compute();

  Vector<6> mu = wls.get_mu();

  //if(limityrot_)
  //  mu[4] = 0.0;

  SE3<double> se3_cur;
  se3_cur = se3_prev * SE3<double>::exp(mu);

  Matrix<3> rot = se3_cur.get_rotation().get_matrix();
  Vector<3> trans = se3_cur.get_translation();

  pose_cur(0,0) = rot(0,0);  
  pose_cur(0,1) = rot(0,1);  
  pose_cur(0,2) = rot(0,2);  
  pose_cur(1,0) = rot(1,0);  
  pose_cur(1,1) = rot(1,1);  
  pose_cur(1,2) = rot(1,2);  
  pose_cur(2,0) = rot(2,0);  
  pose_cur(2,1) = rot(2,1);  
  pose_cur(2,2) = rot(2,2);
  pose_cur(0,3) = trans[0];  
  pose_cur(1,3) = trans[1];  
  pose_cur(2,3) = trans[2];
  pose_cur(3,0) = 0;  
  pose_cur(3,1) = 0;  
  pose_cur(3,2) = 0;  
  pose_cur(3,3) = 1;  
}

Vector<6> EdgeTrackingUtil::calcJacobian(const CvPoint3D32f& pts3, const CvPoint2D32f& pts2, 
  const CvPoint2D32f& ptsnv, double ptsd, const SE3<double> &E, const Eigen::Matrix3f& intrinsics)
{
  Vector<4> vpts3; // 3D points
  Vector<3> vpts2; // 2D points
  Vector<2> vptsn; // normal
  Vector<6> J;

  // Initialize values
  vpts3 = makeVector(pts3.x, pts3.y, pts3.z, 1.0);
  vpts2 = makeVector(pts2.x, pts2.y, 1.0);
  vptsn = makeVector(ptsnv.x, ptsnv.y);
  Matrix<2,2> ja_;
  ja_[0][0] = intrinsics(0,0); ja_[0][1] = 0;
  ja_[1][0] = 0; ja_[1][1] = intrinsics(1,1);

  for(int i = 0; i < 6; i++)
  {
    Vector<4> cam_coord = E*vpts3;
    Vector<4> temp = E*SE3<double>::generator_field(i, vpts3);
    Vector<2> temp2 = temp.slice<0,2>()/cam_coord[2] - cam_coord.slice<0,2>() * (temp[2]/cam_coord[2]/cam_coord[2]);
    J[i] = vptsn*ja_*temp2; // Jc is not required in here.
  }

  return J;
}
