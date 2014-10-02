#include "map_localize/MapLocalizer.h"

int main (int argc, char **argv)
{
  ros::init (argc, argv, "map_localize");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  MapLocalizer ml(nh, nh_private);
  ros::spin ();
  return 0;
}
