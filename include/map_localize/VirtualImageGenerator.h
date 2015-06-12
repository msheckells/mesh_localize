#ifndef _VIRTUAL_IMAGE_GENERATOR_
#define _VIRTUAL_IMAGE_GENERATOR_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

class VirtualImageGenerator
{
public:
  virtual cv::Mat GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask) = 0;  
  virtual Eigen::Matrix3f GetK() = 0;
};
#endif
