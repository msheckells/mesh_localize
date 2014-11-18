#include "map_localize/KeyframeContainer.h"
#include "map_localize/ASiftDetector.h"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
KeyframeContainer::KeyframeContainer(Mat image, std::string desc_type, Eigen::Matrix4f tf, Eigen::Matrix3f K) :
  tf(tf),
  K(K),
  img(image)
{
  if(desc_type == "asift")
  {
    ASiftDetector detector;
    detector.detectAndCompute(img, keypoints, descriptors, ASiftDetector::SIFT);
  }
  else if(desc_type == "asurf")
  {
    ASiftDetector detector;
    detector.detectAndCompute(img, keypoints, descriptors, ASiftDetector::SURF);
  }  
  else if(desc_type == "orb")
  {
    ORB orb(1000);
    Mat mask(img.rows, img.cols, CV_8U, Scalar(255));
    orb(img, mask, keypoints, descriptors);
  }
  else if(desc_type == "surf")
  {
    SurfFeatureDetector detector;
    detector.detect(img, keypoints);

    SurfDescriptorExtractor extractor;
    extractor.compute(img, keypoints, descriptors);
    std::cout << "kps: " << keypoints.size() << std::endl; 
  }
  else if(desc_type == "surf_gpu")
  {
    gpu::SURF_GPU surf_gpu;    
    gpu::GpuMat kps_gpu, mask_gpu, img_gpu(image);
    surf_gpu(img_gpu, mask_gpu, kps_gpu, descriptors_gpu);
    surf_gpu.downloadKeypoints(kps_gpu, keypoints);
  }
}

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, Eigen::Matrix3f K, std::vector<KeyPoint>& keypoints, Mat& descriptors) :
  tf(tf),
  K(K),
  img(image),
  keypoints(keypoints),
  descriptors(descriptors)
{
}

Mat KeyframeContainer::GetImage()
{
  return img;
}

gpu::GpuMat KeyframeContainer::GetGPUDescriptors()
{
  return descriptors_gpu;
}

Mat KeyframeContainer::GetDescriptors()
{
  return descriptors;
}

std::vector<KeyPoint> KeyframeContainer::GetKeypoints()
{
  return keypoints;
}

Eigen::Matrix4f KeyframeContainer::GetTf()
{
  return tf;
}

Eigen::Matrix3f KeyframeContainer::GetK()
{
  return K;
}

