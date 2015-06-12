#ifndef _DEPTH_FEATURE_MATCH_LOCALIZER_H_
#define _DEPTH_FEATURE_MATCH_LOCALIZER_H_

#include "MonocularLocalizer.h"
#include "KeyframeMatch.h"
#include "KeyframeContainer.h"

class DepthFeatureMatchLocalizer : public MonocularLocalizer
{
  struct KeyframePositionSorter
  {
    Eigen::Matrix4f currentPose;
    KeyframePositionSorter(Eigen::Matrix4f pose): currentPose(pose) {};
    
    bool operator()(KeyframeContainer* kfc1, KeyframeContainer* kfc2)
    {
      return (kfc1->GetTf().block<3,1>(0,3)-currentPose.block<3,1>(0,3)).norm() < (kfc2->GetTf().block<3,1>(0,3)-currentPose.block<3,1>(0,3)).norm();
    }
  };

public:

  DepthFeatureMatchLocalizer(const std::vector<KeyframeContainer*>& train, std::string desc_type = "surf", bool show_matches = false);
  virtual bool localize(const cv::Mat& img, const cv::Mat& K, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess = NULL);

private:

  std::vector< KeyframeMatch > FindImageMatches(KeyframeContainer* img, int k, Eigen::Matrix4f* pose_guess = NULL, unsigned int search_bound = 0);

  std::vector<KeyframeContainer*> keyframes; 
  bool show_matches;
  std::string desc_type;
};

#endif
