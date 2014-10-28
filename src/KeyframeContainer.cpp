#include "map_localize/KeyframeContainer.h"
#include "map_localize/ASiftDetector.h"

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, Eigen::Matrix3f K, int minHessian) :
  tf(tf),
  K(K),
  img(image)
{
  //SurfFeatureDetector detector(minHessian);
  //SiftFeatureDetector detector;
  //GoodFeaturesToTrackDetector detector(1500, 0.01, 1.0);
  //detector.detect(img, keypoints);
  
  //SurfDescriptorExtractor extractor;
  //SiftDescriptorExtractor extractor;
  //extractor.compute(img, keypoints, descriptors);

  ASiftDetector detector;
  detector.detectAndCompute(img, keypoints, descriptors);  
}

KeyframeContainer::KeyframeContainer(Mat image, Eigen::Matrix4f tf, Eigen::Matrix3f K, std::vector<KeyPoint>& keypoints, Mat& descriptors, int minHessian) :
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

