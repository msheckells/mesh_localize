#ifndef _GAZEBOLOCALIZER_H_
#define _GAZEBOLOCALIZER_H_
/* This file is for localization using a textured map in Gazebo and looking at a camera.
 * The basic idea is that, we want to minimize the error between the camera position in gazebo with that of real world
 * by minimizing the error between the image taken by gazebo and the real image
*/

#include <vector>
#include <ros/ros.h>
#include "KeyframeContainer.h"
#include "KeyframeMatch.h"
#include <gazebo_msgs/SetModelState.h>
#include <sensor_msgs/Image.h>
//Cv stuff
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


class GazeboLocalizer
{
	public:
		GazeboLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);
		~GazeboLocalizer();

	private:

		//  std::vector<KeyframeContainer*> keyframes;
		gazebo_msgs::SetModelState modelstate_req;
		ros::ServiceClient modelclient;

		//sensor_msgs::Image curr_img;
		sensor_msgs::Image goal_img;
		ros::Subscriber imagesub;

		ros::NodeHandle nh;
		ros::NodeHandle nh_private;

		image_transport::Publisher pub_kf_;
		boost::shared_ptr<image_transport::ImageTransport> it_;

		//Matcher
		FlannBasedMatcher matcher;

		//Functions:
		void imgCallback(const sensor_msgs::ImageConstPtr &input);
		double FindImageError(KeyframeContainer* img1, KeyframeContainer* img2, int &numberofmatches);
};

#endif
