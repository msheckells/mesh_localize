#include "map_localize/KeyframeContainer.h"

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, int minHessian) :
  tf(tf),
  img(image)
{
  //SurfFeatureDetector detector(minHessian);
  //SiftFeatureDetector detector;
  GoodFeaturesToTrackDetector detector;
  detector.detect(img, keypoints);
  
  //SurfDescriptorExtractor extractor;
  SiftDescriptorExtractor extractor;
  extractor.compute(img, keypoints, descriptors);
}

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, std::vector<KeyPoint>& keypoints, Mat descriptors, int minHessian) :
  tf(tf),
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
