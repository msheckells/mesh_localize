#ifndef _KEYFRAMECONTAINER_H_
#define _KEYFRAMECONTAINER_H_

#include "map_localize/CameraContainer.h"

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
  KeyframeContainer(Mat img, std::string desc_type = "asurf");
  KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors);
  KeyframeContainer(CameraContainer* cc, std::string desc_type = "asurf");
  KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors);
 ~KeyframeContainer();
  KeyframeContainer(const KeyframeContainer&);
  
  Mat GetImage();
  Mat GetDescriptors();
  gpu::GpuMat GetGPUDescriptors();
  std::vector<KeyPoint> GetKeypoints();
  Eigen::Matrix4f GetTf();
  Eigen::Matrix3f GetK();
private:

  void ExtractFeatures(std::string desc_type);

  CameraContainer* cc;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  gpu::GpuMat descriptors_gpu;

  bool delete_cc;
};

#endif
