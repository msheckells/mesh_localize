#include "mesh_localize/DepthFeatureMatchLocalizer.h"
#include "mesh_localize/PnPUtil.h"

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

DepthFeatureMatchLocalizer::DepthFeatureMatchLocalizer(const std::vector<KeyframeContainer*>& train,
  std::string desc_type, bool show_matches, int min_inliers, double max_reproj_error,
  double ratio_test_thresh)
  : keyframes(train), desc_type(desc_type), show_matches(show_matches), min_inliers(min_inliers),
    max_reproj_error(max_reproj_error), ratio_test_thresh(ratio_test_thresh)
{
  namedWindow( "Match", WINDOW_NORMAL );
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
  double bestReprojError;
  for(int i = 0; i < matches.size(); i++)
  {
    if(matches[i].matchKps1.size() >= 5)
    { 
      std::vector<int> inlierIdx;
      Eigen::Matrix4f vimgTf = matches[i].kfc->GetTf();
      std::vector<Point3f> matchPts3d = PnPUtil::BackprojectPts(matches[i].matchPts2, vimgTf, matches[i].kfc->GetK(), matches[i].kfc->GetDepth());  
    
      /** Debugging Backproject **
      Eigen::Matrix4f vimgTfI = vimgTf.inverse();
      Mat distcoeffcvPnp = (Mat_<double>(4,1) << 0, 0, 0, 0);
      Mat Rvec, t;
      Mat Rtest = (Mat_<double>(3,3) << vimgTfI(0,0), vimgTfI(0,1), vimgTfI(0,2),
                                     vimgTfI(1,0), vimgTfI(1,1), vimgTfI(1,2),
                                     vimgTfI(2,0), vimgTfI(2,1), vimgTfI(2,2));
      Rodrigues(Rtest, Rvec);
      t = (Mat_<double>(3,1) << vimgTfI(0,3), vimgTfI(1,3), vimgTfI(2,3));
      std::vector<Point2f> reprojPts;
      projectPoints(matchPts3d, Rvec, t, Kcv, distcoeffcvPnp, reprojPts);
      for(int j = 0; j < reprojPts.size(); j++)
      {
        std::cout << j << ": " << reprojPts[j] << " " << matches[i].matchPts2[j] << std::endl;
      }    
      /******/

#if 0
      Mat depth_im;
      double min_depth, max_depth;
      minMaxLoc(matches[i].kfc->GetDepth(), &min_depth, &max_depth);    
      //std::cout << "min_depth=" << min_depth << " max_depth=" << max_depth << std::endl;
      matches[i].kfc->GetDepth().convertTo(depth_im, CV_8U, 255.0/(max_depth-min_depth), 0);// -min_depth*255.0/(max_depth-min_depth));
      namedWindow( "Global Depth", WINDOW_NORMAL );// Create a window for display.
      imshow( "Global Depth", depth_im ); 
#endif
      if(matchPts3d.size() == 0 || matches[i].matchPts1.size() == 0)
        continue;

      Eigen::Matrix4f tf_ransac;
      double reprojError;
      if(!PnPUtil::RansacPnP(matchPts3d, matches[i].matchPts1, Kcv, vimgTf.inverse(), tf_ransac, inlierIdx, &reprojError))
      {
        continue;
      }
      std::cout << "KF " << i << " reproj error: " << reprojError << " #inliers: " << inlierIdx.size()
        << std::endl;
      if(inlierIdx.size() < min_inliers || reprojError >= max_reproj_error)
      {
        continue;
      }

      //std::cout << "Match K=" << std::endl << matches[i].kfc->GetK() << std::endl;
      //std::cout << "Image K=" << std::endl << Kcv << std::endl;
      if(inlierIdx.size() > bestInliers.size());
      {
        bestMatch = i;
        bestInliers = inlierIdx;
        *pose = tf_ransac.inverse();
        bestReprojError = reprojError;
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
    std::cout << "DepthFeatureMatchLocalizer: Found consistent match (" << bestInliers.size() << " inliers).  Avg reproj error: " << bestReprojError << std::endl;
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
  const double matchRatio = ratio_test_thresh;;
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
    if(keyframes[i]->GetDescriptors().rows == 0 || keyframes[i]->GetDescriptors().cols == 0)
      continue;
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
