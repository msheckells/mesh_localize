#include "map_localize/KeyframeContainer.h"

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, int minHessian) :
  img(image),
  tf(tf)
{
  SurfFeatureDetector detector(minHessian);
  detector.detect(img, keypoints);
  
  SurfDescriptorExtractor extractor;
  extractor.compute(img, keypoints, descriptors);
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
