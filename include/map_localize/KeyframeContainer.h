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
  KeyframeContainer(Mat img, std::string desc_type = "asurf", bool extract_now = true);
  KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors);
  KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors, Mat& depth);
  KeyframeContainer(CameraContainer* cc, std::string desc_type = "asurf");
  KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors);
  KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors, Mat& depth);
 ~KeyframeContainer();
  KeyframeContainer(const KeyframeContainer&);
  
  Mat GetImage();
  Mat GetDescriptors();
  Mat GetDepth();
  gpu::GpuMat GetGPUDescriptors();
  std::vector<KeyPoint> GetKeypoints();
  Eigen::Matrix4f GetTf();
  Eigen::Matrix3f GetK();

  void ExtractFeatures();
  void SetMask(Mat new_mask);
private:

  void ExtractFeatures(std::string desc_type);

  CameraContainer* cc;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  gpu::GpuMat descriptors_gpu;
  Mat depth; //May not be used
  Mat mask;
  string desc_type;

  bool delete_cc;
  bool has_depth;
};

#endif
