#ifndef _MONOCULAR_LOCALIZER_H_
#define _MONOCULAR_LOCALIZER_H_

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

class MonocularLocalizer
{
public:
  virtual bool localize(const cv::Mat& img, const cv::Mat& K, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess = NULL) = 0; 
};

#endif
