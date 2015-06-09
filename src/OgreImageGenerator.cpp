#include "map_localize/OgreImageGenerator.h"

using namespace cv;

OgreImageGenerator::OgreImageGenerator()
{
}

Mat OgreImageGenerator::GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask)
{
  return Mat();
}  

