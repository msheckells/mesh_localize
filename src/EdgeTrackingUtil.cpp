#include "mesh_localize/EdgeTrackingUtil.h"
#include "mesh_localize/PnPUtil.h"
#include "TooN/TooN.h"
#include "TooN/SVD.h"       // for SVD
#include "TooN/so3.h"       // for special orthogonal group
#include "TooN/se3.h"       // for special Euclidean group
#include "TooN/wls.h"       // for weighted least square
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <boost/thread.hpp>


using namespace TooN;
using namespace cv;
  
bool EdgeTrackingUtil::show_debug = false;
bool EdgeTrackingUtil::autotune_canny = true;
double EdgeTrackingUtil::canny_high_thresh = 180;
double EdgeTrackingUtil::canny_low_thresh = 60;
double EdgeTrackingUtil::dmax = 15;
double EdgeTrackingUtil::canny_sigma = .33;

bool EdgeTrackingUtil::withinOri(float o1, float o2, float oth)
{
  // both o1 and o2 are in degree
  float diff_o = o1 - o2;
  while(diff_o < -90.f)
    diff_o += 180.f;

  while(diff_o > 90.f)
    diff_o -= 180.f;

  if(diff_o > -oth && diff_o < oth)
    return true;
  else
    return false;
}

void EdgeTrackingUtil::calcImageGradientDirection(Mat& dst, const Mat& src, const std::vector<Point>& pts)
{
  dst = Mat(src.rows, src.cols, CV_32F, Scalar(0)); 
  std::vector<double> egrads = calcImageGradientDirection(src, pts);
  for(int i = 0; i < egrads.size(); i++)
  {
    dst.at<float>(pts[i].y, pts[i].x) = egrads[i];
  }
}

std::vector<double> EdgeTrackingUtil::calcImageGradientDirection(const Mat& src, const std::vector<Point>& pts)
{
  std::vector<double> grad_dirs(pts.size());
  for(int i = 0; i < pts.size(); i++)
  {
    Point pt = pts[i];
    if((pt.x - 1) < 0 || (pt.x+1) >= src.cols || (pt.y-1) < 0 || (pt.y+1) >= src.rows)
    {
      grad_dirs[i] = 0;
    }
    else
    {
      double grad_x = -3*src.at<uchar>(pt.y-1,pt.x-1) +  3*src.at<uchar>(pt.y-1,pt.x+1)
              -10*src.at<uchar>(pt.y,pt.x-1)   + 10*src.at<uchar>(pt.y,pt.x+1) 
               -3*src.at<uchar>(pt.y+1,pt.x-1)   + 3*src.at<uchar>(pt.y+1,pt.x+1);
      double grad_y = -3*src.at<uchar>(pt.y-1,pt.x-1) -10*src.at<uchar>(pt.y-1,pt.x) -3*src.at<uchar>(pt.y-1,pt.x+1)
                      +3*src.at<uchar>(pt.y+1,pt.x-1) +10*src.at<uchar>(pt.y+1,pt.x) +3*src.at<uchar>(pt.y+1,pt.x+1);
      grad_dirs[i] = atan2(grad_y, grad_x);
    }
  }
  return grad_dirs;
}

//SLOW!!!
void EdgeTrackingUtil::calcImageGradientDirection(Mat& dst, const Mat& src)
{
  dst = Mat(src.rows, src.cols, CV_32F, Scalar(0));

  Mat grad_x, grad_y;
  /// Gradient X
  cv::Sobel(src, grad_x, CV_32F, 1, 0, 3);

  /// Gradient Y
  cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);
  
  for(int i = 0; i < src.rows; i++)
  {
    for(int j = 0; j < src.cols; j++)
    {
      dst.at<float>(i,j) = atan2(grad_y.at<float>(i,j), grad_x.at<float>(i,j));
      //std::cout << "(" << i << ", " << j << ") " << dst.at<float>(i,j) << " " << grad_y.at<float>(i,j) << " " << grad_x.at<float>(i,j) << std::endl;
    }
  }
}

void EdgeTrackingUtil::drawLines(Mat& dst, const Mat& src, const std::vector<cv::Vec4i>& lines)
{
  if(src.type() == CV_8UC1)
  {
    cvtColor(src, dst, CV_GRAY2RGB);
  }
  else
  {
    src.copyTo(dst);
  }
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0));
  }
} 

void EdgeTrackingUtil::drawEdgeMatching(Mat& dst, const Mat& src, const vector<EdgeTrackingUtil::SamplePoint>& sps)
{
  if(src.type() == CV_8UC1)
  {
    cvtColor(src, dst, CV_GRAY2RGB);
  }
  else
  {
    dst = src.clone();
  }
  for(int i = 0; i < sps.size(); i++)
  {
     line(dst, Point(sps[i].coord2.x, sps[i].coord2.y), Point(sps[i].edge_pt2.x, sps[i].edge_pt2.y),
       CV_RGB(255, 0, 0));
  }
  for(int i = 0; i < sps.size(); i++)
  {
     circle(dst, Point(sps[i].coord2.x, sps[i].coord2.y), 1, CV_RGB(0, 255, 0));
  }
}

void EdgeTrackingUtil::drawGradientLines(Mat& dst, const Mat& src, const std::vector<Point>& edges, const std::vector<double>& gdir)
{
  if(src.type() == CV_8UC1)
  {
    cvtColor(src, dst, CV_GRAY2RGB);
  }
  else
  {
    dst = src.clone();
  }
  for(int i = 0; i < edges.size(); i++)
  {
    float gd = gdir[i];
    Point p1(edges[i].x - 5*cos(gd), edges[i].y - 5*sin(gd));
    Point p2(edges[i].x + 5*cos(gd), edges[i].y + 5*sin(gd));
    line(dst, p1, p2, CV_RGB(255, 0, 0));
  }
}

void EdgeTrackingUtil::drawGradientLines(Mat& dst, const Mat& src, const Mat& edges, const Mat& gdir)
{
  if(src.type() == CV_8UC1)
  {
    cvtColor(src, dst, CV_GRAY2RGB);
  }
  else
  {
    dst = src.clone();
  }
  for(int i = 6; i < src.rows-6; i++)
  {
    for(int j = 6; j < src.cols-6; j++)
    {
      if(edges.at<uchar>(i,j) == 255)
      {
        float gd = gdir.at<float>(i,j);
        Point p1(j - 5*cos(gd), i - 5*sin(gd));
        Point p2(j + 5*cos(gd), i + 5*sin(gd));
        line(dst, p1, p2, CV_RGB(255, 0, 0));
      }
    }
  }
}

bool EdgeTrackingUtil::extractEdgeDescriptor(Eigen::VectorXd& desc, const Mat& im, Point pt,
  double edge_dir, unsigned int window_size)
{
  double dx = cos(edge_dir);
  double dy = sin(edge_dir);
  desc.resize(2*window_size+1);
  for(int i = 0; i < window_size+1; i++)
  {
    int x1 = round(pt.x + i*dx); 
    int y1 = round(pt.y + i*dy);
    if(x1 < 0 || x1 >= im.cols || y1 < 0 || y1 >= im.rows)
      return false;

    desc(window_size + i) = im.at<uchar>(y1,x1);
    if( i > 0)
    {
      int x2 = round(pt.x - i*dx); 
      int y2 = round(pt.y - i*dy);
      if(x2 < 0 || x2 >= im.cols || y2 < 0 || y2 >= im.rows)
        return false;
       
      desc(window_size - i) = im.at<uchar>(y2,x2);
    } 
  } 
  double mean = 0;
  for(int i = 0; i < desc.size(); i++)
  {
    mean += desc[i]/desc.size();
  }
  Eigen::VectorXd mean_vec = mean*Eigen::VectorXd::Ones(desc.size());
  desc -= mean_vec;
  desc /= desc.norm();
  return true;
}

double EdgeTrackingUtil::getMedian(const Mat& image, int start_bin)
{
  double m;//=(image.rows*image.cols)/2;
  int bin0=0;
  double med = -1;
  int histSize = 256;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist0;
  cv::calcHist(&image, 1, 0, cv::Mat(), hist0, 1, &histSize, &histRange, uniform, accumulate );

  m = 0;
  for (int i = start_bin; i < 256; i++)
  {
    m+=cvRound(hist0.at<float>(i));
  }
  m/=2;
  for (int i=start_bin; i<256 &&  med<0 ;i++)
  {
    bin0=bin0+cvRound(hist0.at<float>(i));
    if (bin0>m && med<0)
      med=i;
  }

  return med;
}

double EdgeTrackingUtil::ncc(const Eigen::VectorXd& d1, const Eigen::VectorXd& d2)
{
  assert(d1.size() == d2.size());
  return d1.dot(d2);
}

std::vector<EdgeTrackingUtil::SamplePoint> EdgeTrackingUtil::getWindowedEdgeMatches(
  const Mat& vimg,
  const std::vector<Point>& vimg_edge_pts, const std::vector<double>& vimg_edge_dirs, 
  const Mat& kf,
  const Mat& kf_detected_edges, const Mat& kf_edge_dir, const Eigen::Matrix3f vimgK, 
  const Eigen::Matrix3f K, const Mat& vdepth, const Eigen::Matrix4f& vimgTf)
{
  double dmax = 100;//sqrt(kf.cols*kf.cols + kf.rows*kf.rows);
  int window_size = 5;
  Eigen::Matrix3f vimgK_inv = vimgK.inverse();

  //Do 1D search along gradient direction to find closest edge in kf for each edge pt in vimg
  std::vector<EdgeTrackingUtil::SamplePoint> sps;
  for(int i = 0; i < vimg_edge_pts.size(); i++)
  {
    Point pt = vimg_edge_pts[i];
    double p_cam_depth = vdepth.at<float>(int(pt.y), int(pt.x));
    if(p_cam_depth == 0 || p_cam_depth == -1)
      continue;

    float edge_dir = vimg_edge_dirs[i];
    // camera intrinsics may not match virtual intrinsics, so we reproject the virtual point to 
    // the real camera frame
    Eigen::Vector3f p_kf = K*vimgK_inv*Eigen::Vector3f(pt.x,pt.y,1); 
    Eigen::VectorXd vimg_desc;
    
    if(!extractEdgeDescriptor(vimg_desc, vimg, pt, edge_dir, window_size))
      continue;

    Point best_match_pt;
    double best_match_norm = -1;
    for(int d = 0; d < dmax; d++)
    {
      int x_idx = p_kf(0) + d*cos(edge_dir);
      int y_idx = p_kf(1) + d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && 
        withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 20))
      {
        Eigen::VectorXd kf_desc;
        if(!extractEdgeDescriptor(kf_desc, kf, Point(x_idx, y_idx), 
          kf_edge_dir.at<float>(y_idx, x_idx), window_size))
        {
          continue;
        }
        double norm = ncc(kf_desc, vimg_desc);
        if(norm > best_match_norm || best_match_norm == -1)
        {
          best_match_norm = norm;
          best_match_pt = Point(x_idx, y_idx);
        }
      }
      //check other direction
      x_idx = p_kf(0) - d*cos(edge_dir);
      y_idx = p_kf(1) - d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && 
        withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 20))
      {
        Eigen::VectorXd kf_desc;
        if(!extractEdgeDescriptor(kf_desc, kf, Point(x_idx, y_idx), 
          kf_edge_dir.at<float>(y_idx, x_idx), window_size))
        {
          continue;
        }
        double norm = ncc(kf_desc, vimg_desc);
        if(norm > best_match_norm || best_match_norm == -1)
        {
          best_match_norm = norm;
          best_match_pt = Point(x_idx, y_idx);
        }
      } 
    }
    if(best_match_norm > 0 /*&& best_match_norm < sqrt(2*window_size+1)*50*/)
    {
      // Found correspondence
      // Store vimg and KF 2D correspondences
      EdgeTrackingUtil::SamplePoint sp; 
      sp.coord2 = cvPoint2D32f(p_kf(0), p_kf(1)); 
      sp.edge_pt2 = best_match_pt; 
      sp.dist = sqrt(pow(p_kf(0) - best_match_pt.x,2)+pow(p_kf(1) - best_match_pt.y,2));
      sp.dx = -cos(edge_dir);
      sp.dy = -sin(edge_dir);
      sp.nuv = cvPoint2D32f(-cos(edge_dir), -sin(edge_dir));
        
      // Back project vimg point to 3D
      Eigen::Vector3f p_cam = vimgK_inv*Eigen::Vector3f(pt.x,pt.y,1); 
      p_cam *= p_cam_depth/p_cam(2);
      Eigen::Vector4f p_world(p_cam(0), p_cam(1), p_cam(2), 1);
      p_world = vimgTf*p_world;
      sp.coord3 = cvPoint3D32f(p_world(0), p_world(1), p_world(2));    
      sps.push_back(sp);
    }
  }
  return sps;
}

std::vector<EdgeTrackingUtil::SamplePoint> EdgeTrackingUtil::getEdgeMatches(
  const std::vector<Point>& vimg_edge_pts, const std::vector<double>& vimg_edge_dirs, 
  const Mat& kf_detected_edges, const Mat& kf_edge_dir, const Eigen::Matrix3f vimgK, 
  const Eigen::Matrix3f K, const Mat& vdepth, const Eigen::Matrix4f& vimgTf)
{
  Eigen::Matrix3f vimgK_inv = vimgK.inverse();

  //Do 1D search along gradient direction to find closest edge in kf for each edge pt in vimg
  std::vector<EdgeTrackingUtil::SamplePoint> sps;
  for(int i = 0; i < vimg_edge_pts.size(); i++)
  {
    Point pt = vimg_edge_pts[i];
    double p_cam_depth = vdepth.at<float>(int(pt.y), int(pt.x));
    if(p_cam_depth == 0 || p_cam_depth == -1)
      continue;
    
    float edge_dir = vimg_edge_dirs[i];
    // camera intrinsics may not match virtual intrinsics, so we reproject the virtual point to 
    // the real camera frame
    Eigen::Vector3f p_kf = K*vimgK_inv*Eigen::Vector3f(pt.x,pt.y,1); 
    for(int d = 0; d < dmax; d++)
    {
      int x_idx = p_kf(0) + d*cos(edge_dir);
      int y_idx = p_kf(1) + d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 20))
      {
        // Found correspondence
        // Store vimg and KF 2D correspondences
        EdgeTrackingUtil::SamplePoint sp; 
        sp.coord2 = cvPoint2D32f(p_kf(0), p_kf(1)); 
        sp.edge_pt2 = cvPoint2D32f(x_idx, y_idx); 
        sp.dist = sqrt(pow(p_kf(0) - x_idx,2)+pow(p_kf(1) - y_idx,2));
        sp.dx = cos(edge_dir);
        sp.dy = sin(edge_dir);
        sp.nuv = cvPoint2D32f(cos(edge_dir), sin(edge_dir));

        // Back project vimg point to 3D
        Eigen::Vector3f p_cam = vimgK_inv*Eigen::Vector3f(pt.x,pt.y,1); 
        p_cam *= p_cam_depth/p_cam(2);
        Eigen::Vector4f p_world(p_cam(0), p_cam(1), p_cam(2), 1);
        p_world = vimgTf*p_world;
        sp.coord3 = cvPoint3D32f(p_world(0), p_world(1), p_world(2));    
        sps.push_back(sp);
        break;
      }
      //check other direction
      x_idx = p_kf(0) - d*cos(edge_dir);
      y_idx = p_kf(1) - d*sin(edge_dir);
      if(x_idx < 0 || x_idx >= kf_detected_edges.cols || y_idx < 0 || y_idx >= kf_detected_edges.rows)
        continue;
      if(kf_detected_edges.at<uchar>(y_idx, x_idx) == 255 && withinOri(edge_dir*180./M_PI, kf_edge_dir.at<float>(y_idx, x_idx)*180./M_PI, 20))
      {
        // Found correspondence
        // Store vimg and KF 2D correspondences
        EdgeTrackingUtil::SamplePoint sp; 
        sp.coord2 = cvPoint2D32f(p_kf(0), p_kf(1)); 
        sp.edge_pt2 = cvPoint2D32f(x_idx, y_idx); 
        sp.dist = sqrt(pow(p_kf(0) - x_idx,2)+pow(p_kf(1) - y_idx,2));
        sp.dx = -cos(edge_dir);
        sp.dy = -sin(edge_dir);
        sp.nuv = cvPoint2D32f(-cos(edge_dir), -sin(edge_dir));
        
        // Back project vimg point to 3D
        Eigen::Vector3f p_cam = vimgK_inv*Eigen::Vector3f(pt.x,pt.y,1); 
        p_cam *= p_cam_depth/p_cam(2);
        Eigen::Vector4f p_world(p_cam(0), p_cam(1), p_cam(2), 1);
        p_world = vimgTf*p_world;
        sp.coord3 = cvPoint3D32f(p_world(0), p_world(1), p_world(2));    
        sps.push_back(sp);
        break;
      }
    }
  }

  return sps;
}



std::vector<EdgeTrackingUtil::SamplePoint> EdgeTrackingUtil::getEdgeMatches(const Mat& vimg, 
  const Mat& kf, const Eigen::Matrix3f vimgK, const Eigen::Matrix3f K, const Mat& vdepth, 
  const Mat& kf_mask, const Eigen::Matrix4f& vimgTf)
{
  std::clock_t start;
  // Get edges from vimg and kf using canny
  Mat kf_detected_edges, vimg_detected_edges;

  start = std::clock();
  double canny_low_thresh1, canny_low_thresh2, canny_high_thresh1, canny_high_thresh2;
  if(autotune_canny)
  {
    double med1 = getMedian(kf,1);
    double med2 = getMedian(vimg, 1); // 1 ignores black background
    canny_low_thresh1 = (1-canny_sigma)*med1;
    canny_low_thresh2 = (1-canny_sigma)*med2;
    canny_high_thresh1 = (1+canny_sigma)*med1;
    canny_high_thresh2 = (1+canny_sigma)*med2;
  }
  else
  {
    canny_low_thresh1 = canny_low_thresh;
    canny_low_thresh2 = canny_low_thresh;
    canny_high_thresh1 = canny_high_thresh;
    canny_high_thresh2 = canny_high_thresh;
  }
  std::cout << "Canny thresh time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  // do both cannys at once since it's slow
  start = std::clock();
  boost::thread canny_thread = boost::thread(Canny, boost::ref(kf), boost::ref(kf_detected_edges), 
    canny_low_thresh1, 
    canny_high_thresh1, 3, false);
    
  Canny(vimg, vimg_detected_edges, canny_low_thresh2, canny_high_thresh2, 3);
  std::cout << "Canny 2 time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  start = std::clock();
  canny_thread.join();
  std::cout << "Canny 1 time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  //start = std::clock();
  //vector<Vec4i> vimg_detected_lines;
  //HoughLinesP(vimg_detected_edges, vimg_detected_lines, 1, CV_PI/180, 30, 30, 10);
  //std::cout << "Lines 1 time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  Mat vimg_edge_pts_mat, kf_edge_pts_mat;
  findNonZero(vimg_detected_edges, vimg_edge_pts_mat);
  findNonZero(kf_detected_edges, kf_edge_pts_mat);
  std::vector<Point> vimg_edge_pts(vimg_edge_pts_mat.total());
  std::vector<Point> kf_edge_pts(kf_edge_pts_mat.total());
  for(int i = 0; i < vimg_edge_pts_mat.total(); i++)
  {
    vimg_edge_pts[i] = vimg_edge_pts_mat.at<Point>(i);
  } 
  for(int i = 0; i < kf_edge_pts_mat.total(); i++)
  {
    if(kf_mask.at<uchar>(kf_edge_pts_mat.at<Point>(i).y, kf_edge_pts_mat.at<Point>(i).x) > 0)
      kf_edge_pts[i] = kf_edge_pts_mat.at<Point>(i);
  } 
  std::cout << "Canny time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  //Get all edge points in vimg and gradients
  start = std::clock();
  Mat kf_edge_dir;
  std::vector<double> vimg_edge_dirs = calcImageGradientDirection(vimg, vimg_edge_pts);
  start = std::clock();
  calcImageGradientDirection(kf_edge_dir, kf, kf_edge_pts);
  //std::vector<double> kf_edge_dirs = calcImageGradientDirection(kf, kf_edge_pts);
  std::cout << "Edge grad time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;

  start = std::clock();
  std::vector<SamplePoint> sps = getEdgeMatches(vimg_edge_pts, vimg_edge_dirs, kf_detected_edges, 
                                   kf_edge_dir, vimgK, K, vdepth, vimgTf);
  //std::vector<SamplePoint> sps = getWindowedEdgeMatches(vimg, vimg_edge_pts, vimg_edge_dirs, 
  //                                 kf,  kf_detected_edges, 
  //                                 kf_edge_dir, vimgK, K, vdepth, vimgTf);
  std::cout << "Edge match time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << std::endl;
  if(show_debug)
  {
    Mat edge_dir_im;
    drawGradientLines(edge_dir_im, vimg_detected_edges, vimg_edge_pts, vimg_edge_dirs); 

    //Mat kf_edge_dir_im;
    //drawGradientLines(kf_edge_dir_im, kf_detected_edges, kf_edge_pts, kf_edge_dirs);

    Mat edge_matching_overlay;
    drawEdgeMatching(edge_matching_overlay, kf, sps);
 
    //Mat vimg_detected_lines_overlay;
    //drawLines(vimg_detected_lines_overlay, vimg, vimg_detected_lines);

    namedWindow( "Query Edges", WINDOW_NORMAL );// Create a window for display.
    imshow( "Query Edges", kf_detected_edges ); 
    namedWindow( "Virtual Edges", WINDOW_NORMAL );// Create a window for display.
    imshow( "Virtual Edges", vimg_detected_edges ); 
    //namedWindow( "Virtual Lines", WINDOW_NORMAL );// Create a window for display.
    //imshow( "Virtual Lines", vimg_detected_lines_overlay ); 
    namedWindow( "Virtual Edge Directions", WINDOW_NORMAL );// Create a window for display.
    imshow( "Virtual Edge Directions", edge_dir_im ); 
    //namedWindow( "Query Edge Directions", WINDOW_NORMAL );// Create a window for display.
    //imshow( "Query Edge Directions", kf_edge_dir_im ); 
    namedWindow( "Edge Matching", WINDOW_NORMAL );// Create a window for display.
    imshow( "Edge Matching", edge_matching_overlay ); 
    waitKey(1);
  }
  return sps;
}  

void EdgeTrackingUtil::getEstimatedPoseIRLS(Eigen::Matrix4f& pose_cur, const Eigen::Matrix4f& pose_pre, const std::vector<SamplePoint>& vSamplePt, const Eigen::Matrix3f& intrinsics)
{
  double alpha_ = 32.;
  // use a numerical non-linear optimization (weighted least square) to find pose (P)

  TooN::Matrix<3,3, double> R;
  TooN::Vector<3> t;
  for(int r=0; r<3; r++)
  {
    t[r] = pose_pre(r,3);
    for(int c=0; c<3; c++)
    {
      R(r,c) = pose_pre(r,c);
    }
  }
  SE3<double> se3_prev(SO3<double>(R),t);

  WLS<6> wls;
  //wls.add_prior(1e1);
  for(int i=0; i<int(vSamplePt.size()); i++)
  {
    if(vSamplePt[i].dist < dmax)
    {
      // INVERSE 1/(alpha_ + dist)
      wls.add_mJ(
        vSamplePt[i].dist,
        calcJacobian(vSamplePt[i].coord3, vSamplePt[i].coord2, vSamplePt[i].nuv, vSamplePt[i].dist, se3_prev, intrinsics),
        1.0/(static_cast<double>(alpha_) + (vSamplePt[i].dist>0 ? vSamplePt[i].dist : -vSamplePt[i].dist))
      );
    }
  }

  wls.compute();

  TooN::Vector<6> mu = wls.get_mu();

  //if(limityrot_)
  //  mu[4] = 0.0;

  SE3<double> se3_cur;
  se3_cur = se3_prev * SE3<double>::exp(mu);

  TooN::Matrix<3> rot = se3_cur.get_rotation().get_matrix();
  TooN::Vector<3> trans = se3_cur.get_translation();

  pose_cur(0,0) = rot(0,0);  
  pose_cur(0,1) = rot(0,1);  
  pose_cur(0,2) = rot(0,2);  
  pose_cur(1,0) = rot(1,0);  
  pose_cur(1,1) = rot(1,1);  
  pose_cur(1,2) = rot(1,2);  
  pose_cur(2,0) = rot(2,0);  
  pose_cur(2,1) = rot(2,1);  
  pose_cur(2,2) = rot(2,2);
  pose_cur(0,3) = trans[0];  
  pose_cur(1,3) = trans[1];  
  pose_cur(2,3) = trans[2];
  pose_cur(3,0) = 0;  
  pose_cur(3,1) = 0;  
  pose_cur(3,2) = 0;  
  pose_cur(3,3) = 1;  
}

TooN::Vector<6> EdgeTrackingUtil::calcJacobian(const CvPoint3D32f& pts3, const CvPoint2D32f& pts2, 
  const CvPoint2D32f& ptsnv, double ptsd, const SE3<double> &E, const Eigen::Matrix3f& intrinsics)
{
  TooN::Vector<4> vpts3; // 3D points
  TooN::Vector<3> vpts2; // 2D points
  TooN::Vector<2> vptsn; // normal
  TooN::Vector<6> J;

  // Initialize values
  vpts3 = makeVector(pts3.x, pts3.y, pts3.z, 1.0);
  vpts2 = makeVector(pts2.x, pts2.y, 1.0);
  vptsn = makeVector(ptsnv.x, ptsnv.y);
  TooN::Matrix<2,2> ja_;
  ja_[0][0] = intrinsics(0,0); ja_[0][1] = 0;
  ja_[1][0] = 0; ja_[1][1] = intrinsics(1,1);

  for(int i = 0; i < 6; i++)
  {
    TooN::Vector<4> cam_coord = E*vpts3;
    TooN::Vector<4> temp = E*SE3<double>::generator_field(i, vpts3);
    TooN::Vector<2> temp2 = temp.slice<0,2>()/cam_coord[2] - cam_coord.slice<0,2>() * (temp[2]/cam_coord[2]/cam_coord[2]);
    J[i] = vptsn*ja_*temp2; // Jc is not required in here.
  }

  return J;
}

void EdgeTrackingUtil::getEstimatedPosePnP(Eigen::Matrix4f& pose_cur, const Eigen::Matrix4f& pose_pre,
  const std::vector<SamplePoint>& vSamplePt, const cv::Mat& intrinsics)
{
  double pnpReprojError;
  std::vector<int> inlierIdx;
  Eigen::Matrix<float, 6, 6> cov;
  std::vector<Point3f> matchPts3d(vSamplePt.size());  
  std::vector<Point2f> matchPts(vSamplePt.size());  

  for(int i = 0; i < vSamplePt.size(); i++)
  {
    matchPts3d[i] = vSamplePt[i].coord3;
    matchPts[i] = vSamplePt[i].edge_pt2;
  }

  PnPUtil::RansacPnP(matchPts3d, matchPts, intrinsics, pose_pre, pose_cur, inlierIdx, 
    &pnpReprojError, &cov);
}
