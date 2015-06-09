
#ifndef _OGRE_IMAGE_GENERATOR_
#define _OGRE_IMAGE_GENERATOR_

#include "map_localize/VirtualImageGenerator.h"

class OgreImageGenerator : public VirtualImageGenerator
{
public:
  OgreImageGenerator();
  virtual cv::Mat GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask);  

};
#endif
