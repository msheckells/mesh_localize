#ifndef _FABMAP_LOCALIZER_H_
#define _FABMAP_LOCALIZER_H_

#include "MonocularLocalizer.h"
#include "KeyframeMatch.h"
#include "KeyframeContainer.h"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

class FABMAPLocalizer : public MonocularLocalizer
{
  struct CameraPositionSorter
  {
    Eigen::Matrix4f currentPose;
    CameraPositionSorter(Eigen::Matrix4f pose): currentPose(pose) {};
    
    bool operator()(CameraContainer* kfc1, CameraContainer* kfc2)
    {
      return (kfc1->GetTf().block<3,1>(0,3)-currentPose.block<3,1>(0,3)).norm() < (kfc2->GetTf().block<3,1>(0,3)-currentPose.block<3,1>(0,3)).norm();
    }
  };

public:

  FABMAPLocalizer(const std::vector<CameraContainer*>& train, std::string descriptor_type, bool show_matches = false, bool load = false, std::string filename = "");
  virtual bool localize(const cv::Mat& img, const cv::Mat& K, Eigen::Matrix4f* pose, Eigen::Matrix4f* poseGuess = NULL);
private:

  std::vector<CameraContainer*> keyframes; 
  std::string desc_type;
  bool show_matches;

  Ptr<FeatureDetector> detector;
  Ptr<DescriptorExtractor> extractor;
  Ptr<DescriptorMatcher> matcher;
  Ptr<of2::FabMap> fabmap;
  Ptr<BOWImgDescriptorExtractor> bide;
};

#endif
