#include "mesh_localize/KLTTracker.h"
#include <iostream>

using namespace Eigen;
using namespace cv;

KLTTracker::KLTTracker()
{
  m_nextID = 0;
  m_maxNumberOfPoints = 200;
  m_fastDetector = cv::FastFeatureDetector::create(std::string("FAST"));
}

std::vector<unsigned char> KLTTracker::filterMatchesEpipolarContraint(
  const std::vector<cv::Point2f>& pts1, 
  const std::vector<cv::Point2f>& pts2)
{
  std::vector<unsigned char> status; 
  findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 3, .99, status);
  return status;
}

void KLTTracker::init(const cv::Mat& inputFrame, const cv::Mat& depth, const Eigen::Matrix3f& inputK, 
  const Eigen::Matrix3f& depthK, const Eigen::Matrix4f& inputTf, const cv::Mat& mask)
{
  m_nextPts.clear();
  m_prevPts.clear();
  m_ptIDs.clear();
  m_tracked3dPts.clear();
  m_nextKeypoints.clear();
  m_prevKeypoints.clear();
  m_mask = mask;
  m_nextID = 0;

  m_fastDetector->detect(inputFrame, m_nextKeypoints, m_mask);
  std::random_shuffle(m_nextKeypoints.begin(), m_nextKeypoints.end());
  m_nextKeypoints.resize(m_maxNumberOfPoints < m_nextKeypoints.size() ? 
    m_maxNumberOfPoints : m_nextKeypoints.size());

  for (size_t i=0; i<m_nextKeypoints.size(); i++)
  {
    Eigen::Vector3f hkp(m_nextKeypoints[i].pt.x, m_nextKeypoints[i].pt.y, 1);
    Eigen::Vector3f depth_kp = depthK*inputK.inverse()*hkp;

    double pt_depth = depth.at<float>(int(depth_kp(1)), int(depth_kp(0)));
    if(pt_depth == 0 || pt_depth == -1)
      continue;

    m_prevPts.push_back(m_nextKeypoints[i].pt);
    m_ptIDs.push_back(m_nextID++);

    Eigen::Vector3f backproj = inputK.inverse()*hkp;
    backproj /= backproj(2);    
    backproj *= pt_depth;
    Eigen::Vector4f backproj_h(backproj(0), backproj(1), backproj(2), 1);
    backproj_h = inputTf*backproj_h;
    m_tracked3dPts.push_back(Point3f(backproj_h(0), backproj_h(1), backproj_h(2)));
  }
  inputFrame.copyTo(m_prevImg);
}

//! Processes a frame and returns output image
bool KLTTracker::processFrame(const cv::Mat& inputFrame, cv::Mat& outputFrame, 
  std::vector<cv::Point2f>& pts2d, std::vector<cv::Point3f>& pts3d, std::vector<int>& ptIDs)
{
  pts2d.clear();
  pts3d.clear();
  inputFrame.copyTo(m_nextImg);
  cv::cvtColor(inputFrame, outputFrame, CV_GRAY2BGR);

  if (m_mask.rows != inputFrame.rows || m_mask.cols != inputFrame.cols)
    m_mask.create(inputFrame.rows, inputFrame.cols, CV_8UC1);
  if (m_prevPts.size() > 0)
  {
    cv::calcOpticalFlowPyrLK(m_prevImg, m_nextImg, m_prevPts, m_nextPts, m_status, m_error);
  }
  m_mask = cv::Scalar(255);
  std::vector<cv::Point2f> lkPrevPts, lkNextPts;
  std::vector<cv::Point3f> lk3dPts;
  std::vector<int> lkTrackedPtIDs;
  for (size_t i=0; i<m_status.size(); i++)
  {
    if (m_status[i])
    {
      lkPrevPts.push_back(m_prevPts[i]);
      lkNextPts.push_back(m_nextPts[i]);
      lk3dPts.push_back(m_tracked3dPts[i]);
      lkTrackedPtIDs.push_back(m_ptIDs[i]);
    }
  }
  std::vector<unsigned char> epStatus;
  if(lkPrevPts.size() > 0)
    epStatus = filterMatchesEpipolarContraint(lkPrevPts, lkNextPts);
  std::vector<cv::Point2f> trackedPts;
  std::vector<cv::Point3f> tracked3dPts;
  std::vector<int> trackedPtIDs;
  for (size_t i=0; i<epStatus.size(); i++)
  {
    if (epStatus[i])
    {
      tracked3dPts.push_back(lk3dPts[i]);
      trackedPts.push_back(lkNextPts[i]);
      trackedPtIDs.push_back(lkTrackedPtIDs[i]);
      cv::circle(m_mask, lkPrevPts[i], 15, cv::Scalar(0), -1);
      cv::line(outputFrame, lkPrevPts[i], lkNextPts[i], cv::Scalar(0,250,0));
      cv::circle(outputFrame, lkNextPts[i], 3, cv::Scalar(0,250,0), -1);
      cv::putText(outputFrame, std::to_string(lkTrackedPtIDs[i]), lkNextPts[i], 
        cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255));
      pts2d.push_back(lkNextPts[i]);
      pts3d.push_back(lk3dPts[i]);
      ptIDs.push_back(lkTrackedPtIDs[i]);
    }
  }
  m_tracked3dPts = tracked3dPts;
  m_prevPts = trackedPts;
  m_ptIDs = trackedPtIDs;
  m_nextImg.copyTo(m_prevImg);
  return true;
}
