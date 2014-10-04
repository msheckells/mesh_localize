#include "map_localize/GazeboLocalizer.h"
#include <algorithm>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 

GazeboLocalizer::GazeboLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private):
    nh(nh),
    nh_private(nh_private)
{

	// TODO: make filepath param
	/* Test Matches */
	srand(time(NULL));
	modelclient = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
	it_.reset(new image_transport::ImageTransport(nh));

	//Setup the services and topics to communicate with gazebo
	imagesub = nh.subscribe("/camera/image_raw", 1, &GazeboLocalizer::imgCallback, this);
	pub_kf_  = it_->advertise("image_keyframe",  1);//Keyframe image

	/*
		 Mat test = imread("/home/matt/uav_image_data/run9/frame0332.jpg", CV_LOAD_IMAGE_GRAYSCALE );//This can be the input image to localize wrt in Gazebo
		 KeyframeContainer* kf = new KeyframeContainer(test, Eigen::Matrix4f());
		 std::vector< KeyframeMatch > matches = FindImageMatches(kf, 5);

		 namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
		 imshow( "Query", kf->GetImage() ); 
		 waitKey(0);
		 for(int i = 0; i < matches.size(); i++)
		 {
		 namedWindow( "Match", WINDOW_AUTOSIZE );// Create a window for display.
		 imshow( "Match", matches[i].kfc->GetImage() ); 
		 waitKey(0);
		 }  
	 */
}

void GazeboLocalizer::imgCallback(const sensor_msgs::ImageConstPtr &input)
{
   const cv::Mat image = cv_bridge::toCvShare(input)->image;
	 ROS_INFO("Got New Image");
	 //Do your computation etc here
	 KeyframeContainer kf(image, Eigen::Matrix4f());
	 sensor_msgs::ImagePtr kf_msg = cv_bridge::CvImage(input->header, "bgr8", kf.GetImage()).toImageMsg();
	 pub_kf_.publish(kf_msg);
	 //ROS_INFO("Display New Image");
}

GazeboLocalizer::~GazeboLocalizer()
{
}

double GazeboLocalizer::FindImageError(KeyframeContainer* kf1, KeyframeContainer* kf2, int &numberofmatches)
{
	const double numMatchThresh = 0;//0.16;
	const double matchRatio = 0.8;
	double imgerror = 0;//Image Error
	numberofmatches = 0;
	std::vector< KeyframeMatch > kfMatches;
	////// Matching the Frames ////////
	std::vector < std::vector< DMatch > > matches;
	matcher.knnMatch( kf1->GetDescriptors(), kf2->GetDescriptors(), matches, 2 );

	//std::vector< DMatch > goodMatches;
	//std::vector<Point2f> matchPts1;
	//std::vector<Point2f> matchPts2;
	// Use ratio test to find good keypoint matches
	for(unsigned int j = 0; j < matches.size(); j++)//Can Parallelize this maybe
	{
		if(matches[j][0].distance < matchRatio*matches[j][1].distance)
		{
			numberofmatches++;
			//goodMatches.push_back(matches[j][0]);
			//matchPts1.push_back(img->GetKeypoints()[matches[j][0].queryIdx].pt);
			//matchPts2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx].pt);
			//Compute the error based on the error between feature matches. 
			//If the number of matches is less than some threshold then the upstream function caller will take care of that using the argument
			//imgerror
		}
	}
	//numberofmatches = goodMatches.size();
	return imgerror;
}
