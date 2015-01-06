#include "map_localize/CameraContainer.h"

CameraContainer::CameraContainer(Mat image, Eigen::Matrix4f tf, Eigen::Matrix3f K) :
  tf(tf),
  K(K),
  img(image)
{
}

Mat CameraContainer::GetImage()
{
  return img;
}

Eigen::Matrix4f CameraContainer::GetTf()
{
  return tf;
}

Eigen::Matrix3f CameraContainer::GetK()
{
  return K;
}

