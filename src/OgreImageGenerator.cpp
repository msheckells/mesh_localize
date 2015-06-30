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
  //std::cout << "Kcv=" << std::endl << Kcv << std::endl ;
  Eigen::Matrix3f K;
  K << Kcv.at<double>(0,0), Kcv.at<double>(0,1), Kcv.at<double>(0,2),
       Kcv.at<double>(1,0), Kcv.at<double>(1,1), Kcv.at<double>(1,2),
       Kcv.at<double>(2,0), Kcv.at<double>(2,1), Kcv.at<double>(2,2);
  //std::cout << "K=" << std::endl << K << std::endl;
  return K;
}

Mat OgreImageGenerator::GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask)
{
  double x = pose(0,3);
  double y = pose(1,3);
  double z = pose(2,3);

  Eigen::Quaternionf q(pose.block<3,3>(0,0));
  Mat im;
  vih->getVirtualImageAndDepth(im, depth, x, y, z, q.w(), q.x(), q.y(), q.z());
  //medianBlur(depth, depth, 5);
  //medianBlur(depth, depth, 5);
  mask = Mat(app->getWindowHeight(), app->getWindowWidth(), CV_8U, Scalar(255));

  for(int i = 0; i < app->getWindowHeight(); i++)
  {
    for(int j = 0; j < app->getWindowWidth(); j++)
    {
      if(depth.at<float>(i, j) == 0 || depth.at<float>(i, j) == -1)
      {
        mask.at<uchar>(i, j) = 0;
      }
    }
  }
  return im;
}  
