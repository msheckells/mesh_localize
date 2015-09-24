#ifndef _KEYFRAMECONTAINER_H_
#define _KEYFRAMECONTAINER_H_

#include "CameraContainer.h"

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef MESH_LOCALIZER_ENABLE_GPU
  #include <opencv2/gpu/gpu.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

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
#ifdef MESH_LOCALIZER_ENABLE_GPU
  gpu::GpuMat GetGPUDescriptors();
#endif
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
#ifdef MESH_LOCALIZER_ENABLE_GPU
  gpu::GpuMat descriptors_gpu;
#endif
  Mat depth; //May not be used
  Mat mask;
  string desc_type;

  bool delete_cc;
  bool has_depth;
};

#endif
