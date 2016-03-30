
#ifndef _OGRE_IMAGE_GENERATOR_
#define _OGRE_IMAGE_GENERATOR_

#include "VirtualImageGenerator.h"
#include "object_renderer/virtual_image_handler.h"

class OgreImageGenerator : public VirtualImageGenerator
{
public:
  OgreImageGenerator(std::string resource_path, std::string model_name, double fx = 400,
    double fy = 400, bool use_depth_shader = true);
  virtual cv::Mat GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask);  
  virtual Eigen::Matrix3f GetK();
  double GetWidth();
  double GetHeight();

private:
  CameraRenderApplication* app;
  VirtualImageHandler* vih;
  bool use_depth_shader;
};
#endif
