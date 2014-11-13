#include "map_localize/KeyframeContainer.h"
#include "map_localize/ASiftDetector.h"
#include "opencv2/features2d/features2d.hpp"

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
    OrbFeatureDetector detector;
    detector.detect(img, keypoints);

    OrbDescriptorExtractor extractor;
    extractor.compute(img, keypoints, descriptors);
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

