#ifndef _KEYFRAMECONTAINER_H_
#define _KEYFRAMECONTAINER_H_

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#define _USE_ASIFT_

using namespace cv;

class KeyframeContainer
{
public:
  KeyframeContainer(Mat image, std::string desc_type = "asurf", Eigen::Matrix4f tf = Eigen::Matrix4f(), Eigen::Matrix3f K = Eigen::Matrix3f());
  KeyframeContainer(Mat image, Eigen::Matrix4f tf, Eigen::Matrix3f K, std::vector<KeyPoint>& keypoints, Mat& descriptors);
  
  Mat GetImage();
  Mat GetDescriptors();
  gpu::GpuMat GetGPUDescriptors();
  std::vector<KeyPoint> GetKeypoints();
  Eigen::Matrix4f GetTf();
  Eigen::Matrix3f GetK();
private:

  Eigen::Matrix4f tf;  
  Eigen::Matrix3f K;  
  Mat img;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  gpu::GpuMat descriptors_gpu;
};

#endif
