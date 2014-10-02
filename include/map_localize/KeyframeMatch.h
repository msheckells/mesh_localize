#ifndef _KEYFRAMEMATCH_H_
#define _KEYFRAMEMATCH_H_

#include "KeyframeContainer.h"

class KeyframeMatch
{
public:
  KeyframeMatch(KeyframeContainer* kfc, std::vector< DMatch >& matches, std::vector < Point2f >& matchPts1, std::vector< Point2f >& matchPts2);

  KeyframeContainer* kfc;
  std::vector< DMatch > matches;
  std::vector< Point2f > matchPts1;
  std::vector< Point2f > matchPts2;
  
  // Try sorting by quality of matches?
  bool operator< (const KeyframeMatch& kfm) const
  {
    return (matches.size() > kfm.matches.size());
  }
};

#endif
