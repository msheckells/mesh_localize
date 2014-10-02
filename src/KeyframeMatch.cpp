#include "map_localize/KeyframeMatch.h"

KeyframeMatch::KeyframeMatch(KeyframeContainer* kfc, std::vector< DMatch >& matches, std::vector < Point2f >& matchPts1, std::vector< Point2f >& matchPts2) :
  kfc(kfc),
  matches(matches),
  matchPts1(matchPts1),
  matchPts2(matchPts2)
{
}
