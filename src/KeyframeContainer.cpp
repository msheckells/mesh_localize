#include "map_localize/KeyframeContainer.h"
#include "map_localize/ASiftDetector.h"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

KeyframeContainer::KeyframeContainer(Mat img, std::string desc_type)
{
  cc = new CameraContainer(img);
  delete_cc = true;
  ExtractFeatures(desc_type);
}

KeyframeContainer::KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors) :
  keypoints(keypoints),
  descriptors(descriptors)
{
  cc = new CameraContainer(img);
  delete_cc = true;
}

KeyframeContainer::KeyframeContainer(CameraContainer* cc, std::string desc_type) :
 cc(cc)
{
  delete_cc = false;
  ExtractFeatures(desc_type);
}

KeyframeContainer::KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors) :
  cc(cc),
  keypoints(keypoints),
  descriptors(descriptors)
{
  delete_cc = false;
}

KeyframeContainer::KeyframeContainer(const KeyframeContainer& kfc)
{
  this->keypoints = kfc.keypoints;
  this->descriptors = kfc.descriptors;
  this->cc = new CameraContainer(kfc.cc->GetImage(), kfc.cc->GetTf(), kfc.cc->GetK());
  this->delete_cc = true;
}

KeyframeContainer::~KeyframeContainer()
{
  if(delete_cc)
  {
    delete cc;
  } 
}

void KeyframeContainer::ExtractFeatures(std::string desc_type)
{
  Mat img = cc->GetImage();
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
    //std::cout << "surf kps: " << keypoints.size() << std::endl; 
  }
  else if(desc_type == "surf_gpu")
  {
    gpu::SURF_GPU surf_gpu;    
    gpu::GpuMat kps_gpu, mask_gpu, img_gpu(img);
    surf_gpu(img_gpu, mask_gpu, kps_gpu, descriptors_gpu);
    surf_gpu.downloadKeypoints(kps_gpu, keypoints);
  }

}

Mat KeyframeContainer::GetImage()
{
  return cc->GetImage();
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
  return cc->GetTf();
}

Eigen::Matrix3f KeyframeContainer::GetK()
{
  return cc->GetK();
}

