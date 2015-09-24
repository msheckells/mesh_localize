#include "mesh_localize/KeyframeContainer.h"
#include "mesh_localize/ASiftDetector.h"
#include "opencv2/features2d/features2d.hpp"

#ifdef MESH_LOCALIZER_ENABLE_GPU
  #include <opencv2/gpu/gpu.hpp>
  #include <opencv2/nonfree/gpu.hpp>
#endif

KeyframeContainer::KeyframeContainer(Mat img, std::string desc_type, bool extract_now)
 : desc_type(desc_type), has_depth(false), delete_cc(true)
{
  mask = Mat(img.rows, img.cols, CV_8U, Scalar(255));
  cc = new CameraContainer(img);
  if(extract_now)
    ExtractFeatures(desc_type);
}

KeyframeContainer::KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors) :
  keypoints(keypoints),
  descriptors(descriptors)
{
  cc = new CameraContainer(img);
  mask = Mat(img.rows, img.cols, CV_8U, Scalar(255));
  delete_cc = true;
  has_depth = false;
}

KeyframeContainer::KeyframeContainer(Mat img, std::vector<KeyPoint>& keypoints, Mat& descriptors, Mat& depth) :
  keypoints(keypoints),
  descriptors(descriptors),
  depth(depth)
{
  cc = new CameraContainer(img);
  mask = Mat(img.rows, img.cols, CV_8U, Scalar(255));
  delete_cc = true;
  has_depth = true;
}
KeyframeContainer::KeyframeContainer(CameraContainer* cc, std::string desc_type) :
 cc(cc)
{
  delete_cc = false;
  mask = Mat(cc->GetImage().rows, cc->GetImage().cols, CV_8U, Scalar(255));
  ExtractFeatures(desc_type);
  has_depth = false;
}

KeyframeContainer::KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors) :
  cc(cc),
  keypoints(keypoints),
  descriptors(descriptors)
{
  mask = Mat(cc->GetImage().rows, cc->GetImage().cols, CV_8U, Scalar(255));
  delete_cc = false;
  has_depth = false;
}

KeyframeContainer::KeyframeContainer(CameraContainer* cc, std::vector<KeyPoint>& keypoints, Mat& descriptors, Mat& depth) :
  cc(cc),
  keypoints(keypoints),
  descriptors(descriptors),
  depth(depth)
{
  mask = Mat(cc->GetImage().rows, cc->GetImage().cols, CV_8U, Scalar(255));
  delete_cc = false;
  has_depth = true;
}

KeyframeContainer::KeyframeContainer(const KeyframeContainer& kfc)
{
  this->keypoints = kfc.keypoints;
  this->descriptors = kfc.descriptors;
  this->cc = new CameraContainer(kfc.cc->GetImage(), kfc.cc->GetTf(), kfc.cc->GetK());
  this->delete_cc = true;
  this->has_depth = kfc.has_depth;
  this->depth = kfc.depth;
  this->mask = kfc.mask;
}

KeyframeContainer::~KeyframeContainer()
{
  if(delete_cc)
  {
    delete cc;
  } 
}

void KeyframeContainer::SetMask(Mat new_mask)
{
  assert(cc->GetImage().rows == mask.rows);
  assert(cc->GetImage().cols == mask.cols);
  mask = new_mask;
}

void KeyframeContainer::ExtractFeatures()
{
  ExtractFeatures(desc_type);
}

void KeyframeContainer::ExtractFeatures(std::string desc_type)
{
  Mat img = cc->GetImage();
  if(desc_type == "asift")
  {
    ASiftDetector detector;
    detector.detectAndCompute(img, keypoints, descriptors, mask, ASiftDetector::SIFT);
  }
  else if(desc_type == "asurf")
  {
    ASiftDetector detector;
    detector.detectAndCompute(img, keypoints, descriptors, mask, ASiftDetector::SURF);
  }  
  else if(desc_type == "orb")
  {
    ORB orb(1000, 1.2f, 4);
    orb(img, mask, keypoints, descriptors);
  }
  else if(desc_type == "surf")
  {
    SurfFeatureDetector detector;
    detector.detect(img, keypoints, mask);

    SurfDescriptorExtractor extractor;
    extractor.compute(img, keypoints, descriptors);
    //std::cout << "surf kps: " << keypoints.size() << std::endl; 
  }
#ifdef MESH_LOCALIZER_ENABLE_GPU
  else if(desc_type == "surf_gpu")
  {
    gpu::SURF_GPU surf_gpu;    
    gpu::GpuMat kps_gpu, mask_gpu(mask), img_gpu(img);
    surf_gpu(img_gpu, mask_gpu, kps_gpu, descriptors_gpu);
    surf_gpu.downloadKeypoints(kps_gpu, keypoints);
  }
#endif

}

Mat KeyframeContainer::GetImage()
{
  return cc->GetImage();
}

Mat KeyframeContainer::GetDepth()
{
  if(!has_depth)
  {
    std::cout << "KeyframeContainer: WARNING, DEPTH NOT SET" << std::endl;
  }
  return depth;
}

#ifdef MESH_LOCALIZER_ENABLE_GPU
gpu::GpuMat KeyframeContainer::GetGPUDescriptors()
{
  return descriptors_gpu;
}
#endif

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

