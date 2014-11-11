#ifndef _KEYFRAMEMATCH_H_
#define _KEYFRAMEMATCH_H_

#include "KeyframeContainer.h"

class KeyframeMatch
{
public:
  KeyframeMatch(KeyframeContainer* kfc, std::vector< DMatch >& matches, std::vector< DMatch >& allMatches, std::vector < Point2f >& matchPts1, std::vector< Point2f >& matchPts2, std::vector< KeyPoint > matchKps1, std::vector< KeyPoint > matchKps2);

  KeyframeContainer* kfc;
  std::vector< DMatch > matches;
  std::vector< DMatch > allMatches;
  std::vector< Point2f > matchPts1;
  std::vector< Point2f > matchPts2;
  std::vector< KeyPoint > matchKps1;
  std::vector< KeyPoint > matchKps2;
  
  bool operator< (const KeyframeMatch& kfm) const;
private:
  mutable int tot_dist;  
};

#endif
