#ifndef _KLTTracker_hpp
#define _KLTTracker_hpp

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class KLTTracker 
{
  
public:
  KLTTracker();
  void init(const cv::Mat& inputFrame, const cv::Mat& depth, const Eigen::Matrix3f& inputK, 
    const Eigen::Matrix3f& depthK, const Eigen::Matrix4f& inputTf, const cv::Mat& mask);
  virtual bool processFrame(const cv::Mat& inputFrame, cv::Mat& outputFrame, 
    std::vector<cv::Point2f>& pts2d, std::vector<cv::Point3f>& pts3d, std::vector<int>& ptIDs);
  std::vector<unsigned char> filterMatchesEpipolarContraint(const std::vector<cv::Point2f>& pts1, 
    const std::vector<cv::Point2f>& pts2);
  void filterPointsByIndex(const std::vector<int>& idxs); 

private:
  int m_maxNumberOfPoints;

  cv::Mat m_prevImg;
  cv::Mat m_nextImg;
  cv::Mat m_mask;

  std::vector<cv::Point2f> m_prevPts;
  std::vector<cv::Point2f> m_nextPts;
  std::vector<cv::Point3f> m_tracked3dPts;

  std::vector<cv::KeyPoint> m_prevKeypoints;
  std::vector<cv::KeyPoint> m_nextKeypoints;

  std::vector<int> m_ptIDs;
  int m_nextID;

  std::vector<unsigned char> m_status;
  std::vector<float> m_error;

  cv::Ptr<cv::FeatureDetector> m_fastDetector;
};
#endif
