#include "map_localize/KeyframeMatch.h"

KeyframeMatch::KeyframeMatch(KeyframeContainer* kfc, std::vector< DMatch >& matches, std::vector< DMatch >& allMatches, std::vector < Point2f >& matchPts1, std::vector< Point2f >& matchPts2, std::vector< KeyPoint > matchKps1, std::vector< KeyPoint > matchKps2) :
  kfc(kfc),
  matches(matches),
  allMatches(allMatches),
  matchPts1(matchPts1),
  matchPts2(matchPts2),
  matchKps1(matchKps1),
  matchKps2(matchKps2),
  tot_dist(-1)
{
}

bool KeyframeMatch::operator< (const KeyframeMatch& kfm) const
{
  /** Try sorting by avg match distance? **
  if(tot_dist < 0)
  {
    tot_dist = 0;
    for(int i = 0; i < matches.size(); i++)
    {
      tot_dist += matches[i].distance;
    }
  }
  if(kfm.tot_dist < 0)
  {
    kfm.tot_dist = 0;
    for(int i = 0; i < kfm.matches.size(); i++)
    {
      kfm.tot_dist += kfm.matches[i].distance;
    }
  }
  return tot_dist/double(matches.size()) < kfm.tot_dist/double(kfm.matches.size());
  **/
  return (matches.size() > kfm.matches.size());
}

