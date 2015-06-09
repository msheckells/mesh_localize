#include "map_localize/DepthFeatureMatchLocalizer.h"
#include "map_localize/PnPUtil.h"

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

DepthFeatureMatchLocalizer::DepthFeatureMatchLocalizer(const std::vector<KeyframeContainer*>& train, bool show_matches)
  : keyframes(train), show_matches(show_matches)
{
}

bool DepthFeatureMatchLocalizer::localize(const Mat& img, const Mat& Kcv, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess)
{
  KeyframeContainer* kf = new KeyframeContainer(img, "surf");
  std::vector< KeyframeMatch > matches;

  if(pose_guess)
  { 
    matches = FindImageMatches(kf, 5, pose_guess, keyframes.size()/4);  
  }
  else
  {
    matches = FindImageMatches(kf, 5);  
  }

  if(show_matches)
  {
    for(int i = 0; i < matches.size(); i++)
    {
      namedWindow( "Match", WINDOW_AUTOSIZE );
      imshow("Match", matches[i].kfc->GetImage());
      waitKey(0);
    }
  }

  if(matches[0].matchKps1.size() >= 40)
  { 
    Eigen::Matrix4f vimgTf = matches[0].kfc->GetTf();
    std::vector<Point3f> matchPts3d = PnPUtil::BackprojectPts(matches[0].matchPts2, vimgTf, matches[0].kfc->GetK(), matches[0].kfc->GetDepth());  
    
    Eigen::Matrix4f tf_ransac;
    if(!PnPUtil::RansacPnP(matchPts3d, matches[0].matchPts1, Kcv, vimgTf.inverse(), tf_ransac))
    {
      return false;
    }

    *pose = tf_ransac.inverse();
    return true;
  }
  else
  {
    printf("Match not good enough: only %d match points\n", int(matches[0].matchKps1.size()));
    return false;
  }
}

std::vector< KeyframeMatch > DepthFeatureMatchLocalizer::FindImageMatches(KeyframeContainer* img, int k, Eigen::Matrix4f* pose_guess, unsigned int search_bound)
{
  const double numMatchThresh = 0;//0.16;
  const double matchRatio = 0.7;
  std::vector< KeyframeMatch > kfMatches;

  if(pose_guess)
  {
    if(search_bound >= keyframes.size())
    {
      search_bound = keyframes.size();
    }
    else 
    {
      KeyframePositionSorter kps(*pose_guess);
      std::sort(keyframes.begin(), keyframes.end(), kps);
    }
  }
  else
  {
    search_bound = keyframes.size();
  }

  // Find potential frame matches
  #pragma omp parallel for
  for(unsigned int i = 0; i < search_bound; i++)
  {
    //std::cout << i/double(keyframes.size()) << std::endl;

    FlannBasedMatcher matcher;
    std::vector < std::vector< DMatch > > matches;
    matcher.knnMatch( img->GetDescriptors(), keyframes[i]->GetDescriptors(), matches, 2 );

    std::vector< DMatch > goodMatches;
    std::vector< DMatch > allMatches;
    std::vector<Point2f> matchPts1;
    std::vector<Point2f> matchPts2;
    std::vector<KeyPoint> matchKps1;
    std::vector<KeyPoint> matchKps2;
    
    // Use ratio test to find good keypoint matches
    for(unsigned int j = 0; j < matches.size(); j++)
    {
      allMatches.push_back(matches[j][0]);
      if(matches[j][0].distance < matchRatio*matches[j][1].distance)
      {
        goodMatches.push_back(matches[j][0]);
        matchPts1.push_back(img->GetKeypoints()[matches[j][0].queryIdx].pt);
        matchPts2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx].pt);
        matchKps1.push_back(img->GetKeypoints()[matches[j][0].queryIdx]);
        matchKps2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx]);
      }
    }
    if(goodMatches.size() >= numMatchThresh*matches.size())
    {
      //std:: cout << "Found Match!" << std::endl;
      #pragma omp critical
      {
        kfMatches.push_back(KeyframeMatch(keyframes[i], goodMatches, allMatches, matchPts1, matchPts2, matchKps1, matchKps2));
      }
    }
  }

  k = (kfMatches.size() < k) ? kfMatches.size() : k;
  std::sort(kfMatches.begin(), kfMatches.end());

  return std::vector< KeyframeMatch > (kfMatches.begin(), kfMatches.begin()+k);
}
