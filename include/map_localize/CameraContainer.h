#ifndef _CAMERACONTAINER_H_
#define _CAMERACONTAINER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace cv;

class CameraContainer
{
public:
  CameraContainer(Mat image, Eigen::Matrix4f tf = Eigen::Matrix4f(), Eigen::Matrix3f K = Eigen::Matrix3f());
  
  Mat GetImage();
  Eigen::Matrix4f GetTf();
  Eigen::Matrix3f GetK();
private:

  Eigen::Matrix4f tf;  
  Eigen::Matrix3f K;  
  Mat img;
};

#endif
