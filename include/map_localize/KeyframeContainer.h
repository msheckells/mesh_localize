#ifndef _KEYFRAMECONTAINER_H_
#define _KEYFRAMECONTAINER_H_

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace cv;

class KeyframeContainer
{
public:
  KeyframeContainer(Mat image, Eigen::Matrix4f tf = Eigen::Matrix4f(), int minHessian = 200);
  KeyframeContainer(Mat image, Eigen::Matrix4f tf, std::vector<KeyPoint>& keypoints, Mat descriptors, int minHessian = 200);
  
  Mat GetImage();
  Mat GetDescriptors();
  std::vector<KeyPoint> GetKeypoints();
  Eigen::Matrix4f GetTf();
private:

  Eigen::Matrix4f tf;  
  Mat img;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
};

#endif
