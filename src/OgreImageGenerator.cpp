#include "map_localize/OgreImageGenerator.h"

using namespace cv;

OgreImageGenerator::OgreImageGenerator(std::string resource_path, std::string model_name)
{
  app = new CameraRenderApplication(resource_path);
  std::cout << "Using OGRE resource path " << resource_path << std::endl;
  app->go();
  app->loadModel("model", model_name);

  vih = new VirtualImageHandler(app);
}

Eigen::Matrix3f OgreImageGenerator::GetK()
{
  Mat Kcv = vih->getCameraIntrinsics();
  Eigen::Matrix3f K;
  K << Kcv.at<float>(0,0), Kcv.at<float>(0,1), Kcv.at<float>(0,2),
       Kcv.at<float>(1,0), Kcv.at<float>(1,1), Kcv.at<float>(1,2),
       Kcv.at<float>(2,0), Kcv.at<float>(2,1), Kcv.at<float>(2,2);
  return K;
}

Mat OgreImageGenerator::GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask)
{
  double x = pose(0,3);
  double y = pose(1,3);
  double z = pose(2,3);
  Eigen::Quaternionf q(pose.block<3,3>(0,0));
  std::cout << "pose=" << std::endl << pose << std::endl;
  Mat im = vih->getVirtualImage(x, y, z, q.w(), q.x(), q.y(), q.z());
  //Mat im = vih->getVirtualImage(0, 0, 10, 0, 0, 0);
  depth = vih->getVirtualDepth(x, y, z, q.w(), q.x(), q.y(), q.z());
  //depth = vih->getVirtualDepth(0, 0, 10, 0, 0, 0);
  mask = Mat(app->getWindowHeight(), app->getWindowWidth(), CV_8U, Scalar(255));
  return im;
}  
