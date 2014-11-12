#include "map_localize/ASiftDetector.h"

#include <iostream>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

ASiftDetector::ASiftDetector()
{

}

void ASiftDetector::detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors, ASiftDetector::DescriptorType desc_type)
{
  int h = img.rows;
  int w = img.cols;
  Mat mask(h, w, CV_8UC1, Scalar(255));

  detectAndCompute(img, keypoints, descriptors, mask, desc_type);
}
  
void ASiftDetector::detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors, const Mat& mask, ASiftDetector::DescriptorType desc_type)
{
  keypoints.clear();
  descriptors = Mat(0, 128, CV_32F);
  
  for(int tl = 1; tl < 4/*6*/; tl++)
  {
    double t = pow(2, 0.5*tl);
    int lim = 1.8*t;
    #pragma omp parallel for
    for(int phil = 0; phil < lim; phil ++)
    //for(int phi = 0; phi < 180; phi += 100.0/t /*72.0/t*/)
    {
      double phi = phil*100.0/t;
      std::vector<KeyPoint> kps;
      Mat desc;

      Mat timg, skew_mask, Ai;
      img.copyTo(timg);
      mask.copyTo(skew_mask);
      affineSkew(t, phi, timg, skew_mask, Ai);

#if 0
      Mat img_disp;
      bitwise_and(mask, timg, img_disp);
      namedWindow( "Skew", WINDOW_AUTOSIZE );// Create a window for display.
      imshow( "Skew", img_disp ); 
      waitKey(0);
#endif
      if(desc_type == ASiftDetector::SIFT)
      {
        SiftFeatureDetector detector;
        detector.detect(timg, kps, skew_mask);
  
        SiftDescriptorExtractor extractor;
        extractor.compute(timg, kps, desc);
      }
      else if(desc_type == ASiftDetector::SURF)
      {
        SurfFeatureDetector detector;
        detector.detect(timg, kps, skew_mask);
  
        SurfDescriptorExtractor extractor;
        extractor.compute(timg, kps, desc);
      }

      for(unsigned int i = 0; i < kps.size(); i++)
      {
        Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
        //std::cout << Ai << std::endl;
        //std::cout << Mat(kpt) << std::endl;
        Mat kpt_t = Ai*Mat(kpt);
        //std::cout << "Transformed kp" << std::endl;
        kps[i].pt.x = kpt_t.at<float>(0,0);
        kps[i].pt.y = kpt_t.at<float>(1,0);
      }
      #pragma omp critical
      {
        keypoints.insert(keypoints.end(), kps.begin(), kps.end());
        descriptors.push_back(desc);
      }
      //std::cout << "Added desc " << descriptors.cols << " " << descriptors.rows << " " << desc.cols << " " << desc.rows << std::endl;
    }   
  }
}
   
void ASiftDetector::affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
  int h = img.rows;
  int w = img.cols;

  //mask = Mat(h, w, CV_8UC1, Scalar(255));  

  Mat A = Mat::eye(2,3, CV_32F);

  if(phi != 0.0)
  {
    phi *= M_PI/180.;
    double s = sin(phi);
    double c = cos(phi);
    
    A = (Mat_<float>(2,2) << c, -s, s, c);

    Mat corners = (Mat_<float>(4,2) << 0, 0, w, 0, w, h, 0, h);
    Mat tcorners = corners*A.t();
    Mat tcorners_x, tcorners_y;
    tcorners.col(0).copyTo(tcorners_x);
    tcorners.col(1).copyTo(tcorners_y);
    std::vector<Mat> channels;
    channels.push_back(tcorners_x);
    channels.push_back(tcorners_y);
    merge(channels, tcorners);

    Rect rect = boundingRect(tcorners);
    A =  (Mat_<float>(2,3) << c, -s, -rect.x, s, c, -rect.y);
    
    warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
  }
  if(tilt != 1.0)
  {
    double s = 0.8*sqrt(tilt*tilt-1);
    GaussianBlur(img, img, Size(0,0), s, 0.01);
    resize(img, img, Size(0,0), 1.0/tilt, 1.0, INTER_NEAREST);
    A.row(0) = A.row(0)/tilt;
  }
  if(tilt != 1.0 || phi != 0.0)
  {
    h = img.rows;
    w = img.cols;
    warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
  }
  invertAffineTransform(A, Ai);
}

