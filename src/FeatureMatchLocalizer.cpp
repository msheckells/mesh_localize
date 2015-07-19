#include "mesh_localize/FeatureMatchLocalizer.h"

#include <fstream>

using namespace cv;

FeatureMatchLocalizer::FeatureMatchLocalizer(const std::vector<CameraContainer*>& train, std::string descriptor_type, bool show_matches,  bool load_descriptors, std::string desc_filename)
  : desc_type(descriptor_type), show_matches(show_matches)
{
  std::ifstream desc_file;
  if(load_descriptors)
  {
    printf("Opening keypoints and descriptors...");
    desc_file.open(desc_filename.c_str(), std::ifstream::binary);
    if(!desc_file.is_open())
    {
      printf("Could not open descriptor file %s", desc_filename.c_str());
      return;
    }
    std::cout << "Successfully opened keypoints and descriptors" << std::endl;
  }

  for(int i = 0; i < train.size(); i++)
  {
    KeyframeContainer* kfc;
    if(load_descriptors && desc_filename != "")
    {
      std::vector<KeyPoint> keypoints;
      int size;
      desc_file.read((char*)&size, sizeof(int));
      for(int j = 0; j < size; j++)
      {
        KeyPoint kp;
        desc_file.read((char*)&kp, sizeof(KeyPoint));
        keypoints.push_back(kp);
      }
      int rows;
      int cols;
      int elemSize;
      desc_file.read((char*)&rows, sizeof(int));
      desc_file.read((char*)&cols, sizeof(int));
      desc_file.read((char*)&elemSize, sizeof(int));
          
      unsigned char* data = new unsigned char[rows*cols*elemSize];
      desc_file.read((char*)data, sizeof(unsigned char)*rows*cols*elemSize);
          
      Mat descriptors(Size(cols,rows), CV_32F, data);
         
      kfc = new KeyframeContainer(train[i], keypoints, descriptors);
    }
    else
    {  
      kfc = new KeyframeContainer(train[i], descriptor_type);
    }
    keyframes.push_back(kfc);    
  }
  printf("Successfully loaded images");

  if(load_descriptors)
  {
    desc_file.close();
  }
  else if(desc_filename != "")
  {
    WriteDescriptorsToFile(desc_filename);
  }
}

bool FeatureMatchLocalizer::WriteDescriptorsToFile(std::string filename)
{
  std::ofstream file;
  file.open(filename.c_str(), std::ios::out | std::ios::binary);

  if(!file.is_open())
  {
    printf("Could not open descriptor file %s", filename.c_str());
    return false;
  }

  printf("Saving keypoints and descriptors...");
  for(unsigned int i = 0; i < keyframes.size(); i++)
  {
    int size = keyframes[i]->GetKeypoints().size();
    file.write((char*)&size, sizeof(int));
    for(int j = 0; j < size; j++)
    {
      KeyPoint kp = keyframes[i]->GetKeypoints()[j];
      file.write((char*)&kp, sizeof(KeyPoint));
    }
    int rows = keyframes[i]->GetDescriptors().rows;
    int cols = keyframes[i]->GetDescriptors().cols;
    int elemSize = keyframes[i]->GetDescriptors().elemSize1();
    unsigned char* data = keyframes[i]->GetDescriptors().data;
   
    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));
    file.write((char*)&elemSize, sizeof(int));
    file.write((char*)data, sizeof(unsigned char)*rows*cols*elemSize);
  }

  file.close();
  printf("Successfully saved keypoints and descriptors to %s", filename.c_str());
  return true;
}

bool FeatureMatchLocalizer::localize(const Mat& img, const Mat& K, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess)
{
  KeyframeContainer* kf = new KeyframeContainer(img, desc_type);
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
    *pose = matches[0].kfc->GetTf();
    return true;
  }
  else
  {
    printf("Match not good enough: only %d match points\n", int(matches[0].matchKps1.size()));
    return false;
  }
}

std::vector< KeyframeMatch > FeatureMatchLocalizer::FindImageMatches(KeyframeContainer* img, int k, Eigen::Matrix4f* pose_guess, unsigned int search_bound)
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
