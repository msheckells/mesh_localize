#include "map_localize/OgreImageGenerator.h"

using namespace cv;

OgreImageGenerator::OgreImageGenerator(std::string resource_path)
{
  app = new CameraRenderApplication(resource_path);
  app->go();
  app->loadModel("model", "box.mesh");

  vih = new VirtualImageHandler(app);
}

Mat OgreImageGenerator::GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask)
{
  double x = pose(0,3);
  double y = pose(1,3);
  double z = pose(2,3);
  Eigen::Quaternionf q(pose.block<3,3>(0,0));
  Mat im = vih->getVirtualImage(x, y, z, q.w(), q.x(), q.y(), q.z());
  depth = vih->getVirtualDepth(x, y, z, q.w(), q.x(), q.y(), q.z());
  mask = Mat(app->getWindowHeight(), app->getWindowWidth(), CV_8U, Scalar(0));
  return im;
}  
