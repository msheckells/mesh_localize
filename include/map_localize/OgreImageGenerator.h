
#ifndef _OGRE_IMAGE_GENERATOR_
#define _OGRE_IMAGE_GENERATOR_

#include "map_localize/VirtualImageGenerator.h"
#include "object_renderer/virtual_image_handler.h"

class OgreImageGenerator : public VirtualImageGenerator
{
public:
  OgreImageGenerator(std::string resource_path);
  virtual cv::Mat GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask);  

private:
  CameraRenderApplication* app;
  VirtualImageHandler* vih;
};
#endif
