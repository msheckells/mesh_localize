#ifndef _EDGE_TRACKING_UTIL_H_
#define _EDGE_TRACKING_UTIL_H_

/**
 *  This class is directly adapted from an implementation given by 
 *  Changhyun Choi and Henrik Christensen, the authors of "Real-time 
 *  3D Model-based Tracking Using Edge and Keypoint Features for 
 *  Robotic Manipulation" (ICRA 2010)
 */
#include <opencv/cv.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "TooN/se3.h"       // for special Euclidean group

class EdgeTrackingUtil
{
public:
  struct SamplePoint
  {
    CvPoint3D32f coord3;        // Coordinate of a visible 3D sampling point
    CvPoint2D32f coord2;        // Coordinate of a visible 2D sampling point
    CvPoint2D32f nuv;           // Normal unit vector of a visible 2D sampling point
    unsigned char nidx;         // Normal index of a visible 2D sampling point (0, 1, 2, 3)
    double dist;                // Normal distance of a visible 3D sampling point
    double normal_ang;          // Normal angle (4 ways)
    double normal_ang_deg;      // Normal angle in degree (accurate)
    double dx;                  // Normal unit vector (x)
    double dy;                  // Normal unit vector (y)
    CvPoint2D32f edge_pt2;      // corresponding edge coordinate
    int edge_mem;               // Edge membership (0, 1, 2, ... until # of edges -1)
  };
  static void getEstimatedPoseIRLS(Eigen::Matrix4f& pose_cur, const Eigen::Matrix4f& pose_pre, const std::vector<SamplePoint>& vSamplePt, const Eigen::Matrix3f& intrinsics);
  static TooN::Vector<6> calcJacobian(const CvPoint3D32f& pts3, const CvPoint2D32f& pts2, 
    const CvPoint2D32f& ptsnv, double ptsd, const TooN::SE3<double> &E, 
    const Eigen::Matrix3f& intrinsics);
  static std::vector<SamplePoint> getEdgeMatches(const cv::Mat& vimg, const cv::Mat& kf, 
    const Eigen::Matrix3f vimgK, const Eigen::Matrix3f K, const cv::Mat& vdepth, 
    const cv::Mat& kf_mask, const Eigen::Matrix4f& vimgTf);
  static void calcImageGradientDirection(cv::Mat& dst, const cv::Mat& src);
  static void drawGradientLines(cv::Mat& dst, const cv::Mat& src, const cv::Mat& edges, 
    const cv::Mat& gdir);
  static void drawEdgeMatching(cv::Mat& dst, const cv::Mat& src, const std::vector<SamplePoint>& sps);
  static bool withinOri(float o1, float o2, float oth);

  static bool show_debug;
  static double canny_high_thresh;
  static double canny_low_thresh;
};

#endif
