#include "map_localize/DepthFeatureMatchLocalizer.h"
#include "map_localize/PnPUtil.h"

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

DepthFeatureMatchLocalizer::DepthFeatureMatchLocalizer(const std::vector<KeyframeContainer*>& train, std::string desc_type, bool show_matches)
  : keyframes(train), desc_type(desc_type), show_matches(show_matches)
{
  namedWindow( "Match", WINDOW_AUTOSIZE );
}

bool DepthFeatureMatchLocalizer::localize(const Mat& img, const Mat& Kcv, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess)
{
  KeyframeContainer* kf = new KeyframeContainer(img, desc_type);
  std::vector< KeyframeMatch > matches;

  if(pose_guess)
  { 
    matches = FindImageMatches(kf, 10, pose_guess, keyframes.size()/4);  
  }
  else
  {
    matches = FindImageMatches(kf, 10);  
  }

  // Find most geometrically consistent match
  int bestMatch = -1;
  std::vector<int> bestInliers;
  for(int i = 0; i < matches.size(); i++)
  {
    if(matches[i].matchKps1.size() >= 30)
    { 
      std::vector<int> inlierIdx;
      Eigen::Matrix4f vimgTf = matches[i].kfc->GetTf();
      std::vector<Point3f> matchPts3d = PnPUtil::BackprojectPts(matches[i].matchPts2, vimgTf, matches[i].kfc->GetK(), matches[i].kfc->GetDepth());  
    
      Eigen::Matrix4f tf_ransac;
      if(!PnPUtil::RansacPnP(matchPts3d, matches[i].matchPts1, Kcv, vimgTf.inverse(), tf_ransac, inlierIdx))
      {
        continue;
      }

      std::cout << "Match K=" << std::endl << matches[i].kfc->GetK() << std::endl;
      std::cout << "Image K=" << std::endl << Kcv << std::endl;
      if(inlierIdx.size() > bestInliers.size());
      {
        bestMatch = i;
        bestInliers = inlierIdx;
        *pose = tf_ransac.inverse();
      }
    }
  }
  if(bestMatch >= 0)
  {
    if(show_matches)
    {
      std::vector< DMatch > inlierMatches;
      for(int j = 0; j < bestInliers.size(); j++)
      {
        inlierMatches.push_back(matches[bestMatch].matches[bestInliers[j]]);
      }

      Mat img_matches;
      drawMatches(img, kf->GetKeypoints(), matches[bestMatch].kfc->GetImage(), matches[bestMatch].kfc->GetKeypoints(), inlierMatches, img_matches);
      imshow("Match", img_matches);
      waitKey(1);
    }
    std::cout << "DepthFeatureMatchLocalizer: Found consistent match (" << bestInliers.size() << " inliers)" << std::endl;
    std::cout << "PnpPose=" << std::endl << *pose << std::endl;
    std::cout << "MatchPose=" << std::endl << matches[bestMatch].kfc->GetTf() << std::endl;
    return true;
  }
  else
  {
    std::cout << "DepthFeatureMatchLocalizer: no matches are geometrically consistent" << std::endl;
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
