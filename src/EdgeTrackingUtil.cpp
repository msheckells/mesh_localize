#include "map_localize/EdgeTrackingUtil.h"
#include "TooN/TooN.h"
#include "TooN/SVD.h"       // for SVD
#include "TooN/so3.h"       // for special orthogonal group
#include "TooN/se3.h"       // for special Euclidean group
#include "TooN/wls.h"       // for weighted least square
#include <opencv2/highgui/highgui.hpp>


using namespace TooN;
using namespace cv;
  
bool EdgeTrackingUtil::show_debug = false;
double EdgeTrackingUtil::canny_high_thresh = 180;
double EdgeTrackingUtil::canny_low_thresh = 60;

bool EdgeTrackingUtil::withinOri(float o1, float o2, float oth)
{
  // both o1 and o2 are in degree
  float diff_o = o1 - o2;
  while(diff_o < -90.f)
    diff_o += 180.f;

  while(diff_o > 90.f)
    diff_o -= 180.f;

  if(diff_o > -oth && diff_o < oth)
    return true;
  else
    return false;
}

void EdgeTrackingUtil::calcImageGradientDirection(Mat& dst, const Mat& src)
{
  dst = Mat(src.rows, src.cols, CV_32F, Scalar(0));

  Mat grad_x, grad_y;
  /// Gradient X
  cv::Sobel(src, grad_x, CV_32F, 1, 0, 3);

  /// Gradient Y
  cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);
  
  for(int i = 0; i < src.rows; i++)
  {
    for(int j = 0; j < src.cols; j++)
    {
      dst.at<float>(i,j) = atan2(grad_y.at<float>(i,j), grad_x.at<float>(i,j));
      //std::cout << "(" << i << ", " << j << ") " << dst.at<float>(i,j) << " " << grad_y.at<float>(i,j) << " " << grad_x.at<float>(i,j) << std::endl;
    }
  }
}

void EdgeTrackingUtil::drawEdgeMatching(Mat& dst, const Mat& src, const vector<EdgeTrackingUtil::SamplePoint>& sps)
{
  cvtColor(src, dst, CV_GRAY2RGB);
  for(int i = 0; i < sps.size(); i++)
  {
     line(dst, Point(sps[i].coord2.x, sps[i].coord2.y), Point(sps[i].edge_pt2.x, sps[i].edge_pt2.y),
       CV_RGB(255, 0, 0));
  }
  for(int i = 0; i < sps.size(); i++)
  {
     circle(dst, Point(sps[i].coord2.x, sps[i].coord2.y), 1, CV_RGB(0, 255, 0));
  }
}

void EdgeTrackingUtil::drawGradientLines(Mat& dst, const Mat& src, const Mat& edges, const Mat& gdir)
{
  cvtColor(src, dst, CV_GRAY2RGB);
  for(int i = 6; i < src.rows-6; i++)
  {
    for(int j = 6; j < src.cols-6; j++)
    {
      if(edges.at<uchar>(i,j) == 255)
      {
        float gd = gdir.at<float>(i,j);
        Point p1(j - 5*cos(gd), i - 5*sin(gd));
        Point p2(j + 5*cos(gd), i + 5*sin(gd));
        line(dst, p1, p2, CV_RGB(255, 0, 0));
      }
    }
  }
}

std::vector<EdgeTrackingUtil::SamplePoint> EdgeTrackingUtil::getEdgeMatches(const Mat& vimg, const Mat& kf, const Eigen::Matrix3f vimgK, const Eigen::Matrix3f K, const Mat& vdepth, const Mat& kf_mask, const Eigen::Matrix4f& vimgTf)
{
  // Get edges from vimg and kf using canny
  Mat kf_detected_edges, vimg_detected_edges, kf_detected_edges_temp;
  Canny(kf, kf_detected_edges_temp, canny_low_thresh, canny_high_thresh, 3);
  kf_detected_edges_temp.copyTo(kf_detected_edges, kf_mask);
    
  Canny(vimg, vimg_detected_edges, canny_low_thresh, canny_high_thresh, 3);

  //Get all edge points in vimg and gradients
  Mat vimg_edge_dir, kf_edge_dir;
  calcImageGradientDirection(vimg_edge_dir, vimg);
  calcImageGradientDirection(kf_edge_dir, kf);
  Mat edgePts;
  findNonZero(vimg_detected_edges, edgePts);

  //Do 1D search along gradient direction to find closest edge in kf for each edge pt in vimg
  int dmax = 20;
  std::vector<EdgeTrackingUtil::SamplePoint> sps;
  for(int i = 0; i < edgePts.total(); i++)
  {
    Point pt = edgePts.at<Point>(i);
    for(int d = 0; d < dmax; d++)
    {
      float edge_dir = vimg_edge_dir.at<float>(pt.y,pt.x);
      // camera intrinsics may not match virtual intrinsics, so we reproject the virtual point to the real camera frame
      Eigen::Vector3f p_kf = K*vimgK.inverse()*Eigen::Vector3f(pt.x,pt.y,1); 
      p_kf /= p_kf(2);
      
      int x_idx = p_kf(0) + d*cos(edge_dir);
      int y_idx = p_kf(1) + d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 15))
      {
        double p_cam_depth = vdepth.at<float>(int(pt.y), int(pt.x));
        if(p_cam_depth == 0 || p_cam_depth == -1)
          continue;
        // Found correspondence
        // Store vimg and KF 2D correspondences
        EdgeTrackingUtil::SamplePoint sp; 
        sp.coord2 = cvPoint2D32f(p_kf(0), p_kf(1)); 
        sp.edge_pt2 = cvPoint2D32f(x_idx, y_idx); 
        sp.dist = sqrt(pow(p_kf(0) - x_idx,2)+pow(p_kf(1) - y_idx,2));
        sp.dx = cos(edge_dir);
        sp.dy = sin(edge_dir);
        sp.nuv = cvPoint2D32f(cos(edge_dir), sin(edge_dir));

        // Back project vimg point to 3D
        Eigen::Vector3f p_cam = vimgK.inverse()*Eigen::Vector3f(pt.x,pt.y,1); 
        p_cam *= p_cam_depth/p_cam(2);
        Eigen::Vector4f p_world(p_cam(0), p_cam(1), p_cam(2), 1);
        p_world = vimgTf*p_world;
        sp.coord3 = cvPoint3D32f(p_world(0), p_world(1), p_world(2));    
        sps.push_back(sp);
        break;
      }
      //check other direction
      x_idx = p_kf(0) - d*cos(edge_dir);
      y_idx = p_kf(1) - d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 15))
      {
        double p_cam_depth = vdepth.at<float>(int(pt.y), int(pt.x));
        if(p_cam_depth == 0 || p_cam_depth == -1)
          continue;
        // Found correspondence
        // Store vimg and KF 2D correspondences
        EdgeTrackingUtil::SamplePoint sp; 
        sp.coord2 = cvPoint2D32f(p_kf(0), p_kf(1)); 
        sp.edge_pt2 = cvPoint2D32f(x_idx, y_idx); 
        sp.dist = sqrt(pow(p_kf(0) - x_idx,2)+pow(p_kf(1) - y_idx,2));
        sp.dx = -cos(edge_dir);
        sp.dy = -sin(edge_dir);
        sp.nuv = cvPoint2D32f(-cos(edge_dir), -sin(edge_dir));
        
        // Back project vimg point to 3D
        Eigen::Vector3f p_cam = vimgK.inverse()*Eigen::Vector3f(pt.x,pt.y,1); 
        p_cam *= p_cam_depth/p_cam(2);
        Eigen::Vector4f p_world(p_cam(0), p_cam(1), p_cam(2), 1);
        p_world = vimgTf*p_world;
        sp.coord3 = cvPoint3D32f(p_world(0), p_world(1), p_world(2));    
        sps.push_back(sp);
        break;
      }
    }
  }

  if(show_debug)
  {
    Mat edge_dir_im;
    drawGradientLines(edge_dir_im, vimg_detected_edges, vimg_detected_edges, vimg_edge_dir); 

    Mat edge_matching_overlay;
    drawEdgeMatching(edge_matching_overlay, kf, sps);

    namedWindow( "Query Edges", WINDOW_NORMAL );// Create a window for display.
    imshow( "Query Edges", kf_detected_edges ); 
    namedWindow( "Virtual Edges", WINDOW_NORMAL );// Create a window for display.
    imshow( "Virtual Edges", vimg_detected_edges ); 
    namedWindow( "Virtual Edge Directions", WINDOW_NORMAL );// Create a window for display.
    imshow( "Virtual Edge Directions", edge_dir_im ); 
    namedWindow( "Edge Matching", WINDOW_NORMAL );// Create a window for display.
    imshow( "Edge Matching", edge_matching_overlay ); 
    waitKey(1);
  }

  return sps;
}  

void EdgeTrackingUtil::getEstimatedPoseIRLS(Eigen::Matrix4f& pose_cur, const Eigen::Matrix4f& pose_pre, const std::vector<SamplePoint>& vSamplePt, const Eigen::Matrix3f& intrinsics)
{
  double maxd_ = 10.;
  double alpha_ = 64.;
  // use a numerical non-linear optimization (weighted least square) to find pose (P)

  TooN::Matrix<3,3, double> R;
  TooN::Vector<3> t;
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

  TooN::Vector<6> mu = wls.get_mu();

  //if(limityrot_)
  //  mu[4] = 0.0;

  SE3<double> se3_cur;
  se3_cur = se3_prev * SE3<double>::exp(mu);

  TooN::Matrix<3> rot = se3_cur.get_rotation().get_matrix();
  TooN::Vector<3> trans = se3_cur.get_translation();

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

TooN::Vector<6> EdgeTrackingUtil::calcJacobian(const CvPoint3D32f& pts3, const CvPoint2D32f& pts2, 
  const CvPoint2D32f& ptsnv, double ptsd, const SE3<double> &E, const Eigen::Matrix3f& intrinsics)
{
  TooN::Vector<4> vpts3; // 3D points
  TooN::Vector<3> vpts2; // 2D points
  TooN::Vector<2> vptsn; // normal
  TooN::Vector<6> J;

  // Initialize values
  vpts3 = makeVector(pts3.x, pts3.y, pts3.z, 1.0);
  vpts2 = makeVector(pts2.x, pts2.y, 1.0);
  vptsn = makeVector(ptsnv.x, ptsnv.y);
  TooN::Matrix<2,2> ja_;
  ja_[0][0] = intrinsics(0,0); ja_[0][1] = 0;
  ja_[1][0] = 0; ja_[1][1] = intrinsics(1,1);

  for(int i = 0; i < 6; i++)
  {
    TooN::Vector<4> cam_coord = E*vpts3;
    TooN::Vector<4> temp = E*SE3<double>::generator_field(i, vpts3);
    TooN::Vector<2> temp2 = temp.slice<0,2>()/cam_coord[2] - cam_coord.slice<0,2>() * (temp[2]/cam_coord[2]/cam_coord[2]);
    J[i] = vptsn*ja_*temp2; // Jc is not required in here.
  }

  return J;
}
