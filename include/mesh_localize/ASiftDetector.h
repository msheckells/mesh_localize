#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

class ASiftDetector
{
public: 
  enum DescriptorType
  {
    SIFT,
    SURF
  };

  ASiftDetector();

  void detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors, const Mat& mask, DescriptorType desc_type = SURF);
  void detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors, DescriptorType desc_type = SURF);

private:
  void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai);  
};
