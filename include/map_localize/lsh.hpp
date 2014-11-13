/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 Author: Vincent Rabaud
 *********************************************************************/

#ifndef LSH_HPP_
#define LSH_HPP_

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>
#include <stdint.h>

#include <opencv2/features2d/features2d.hpp>

namespace lsh
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** What is stored in an LSH bucket
 */
typedef uint32_t FeatureIndex;
/** The id from which we can get a bucket back in an LSH table
 */
typedef unsigned int BucketKey;

/** A bucket in an LSH table
 */
typedef std::vector<FeatureIndex> Bucket;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** POD for stats about an LSH table
 */
struct LshStats
{
  std::vector<unsigned int> bucket_sizes_;
  size_t n_buckets_;
  size_t bucket_size_mean_;
  size_t bucket_size_median_;
  size_t bucket_size_min_;
  size_t bucket_size_max_;
  size_t bucket_size_std_dev;
  /** Each contained vector contains three value: beginning/end for interval, number of elements in the bin
   */
  std::vector<std::vector<unsigned int> > size_histogram_;
};

/** Overload the << operator for LshStats
 * @param out the streams
 * @param stats the stats to display
 * @return the streams
 */
std::ostream& operator <<(std::ostream& out, const LshStats & stats);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Lsh hash table. As its key is a sub-feature, and as usually
 * the size of it is pretty small, we keep it as a continuous memory array.
 * The value is an index in the corpus of features (we keep it as an unsigned
 * int for pure memory reasons, it could be a size_t)
 */
class LshTable
{
public:
  /** A container of all the feature indices. Optimized for space
   */
  typedef boost::unordered_map<BucketKey, Bucket> BucketsSpace;
  /** A container of all the feature indices. Optimized for speed
   */
  typedef std::vector<Bucket> BucketsSpeed;

  /** Constructor
   * Create the mask and allocate the memory
   * @param feature_size is the size of the feature (considered as a char[])
   * @param key_size is the number of bits that are turned on in the feature
   */
  LshTable(unsigned int feature_size, unsigned int feature_number);

  /** Add a feature to the table
   * @param value the value to store for that feature
   * @param feature the feature itself
   */
  void add(unsigned int value, const cv::Mat & feature);

  /** Get a bucket given the key
   * @param key
   * @return
   */
  const Bucket * getBucketFromKey(BucketKey key) const;

  /** Print the mask, as hex string
   */
  std::ostream& printMask(std::ostream& out) const;

  LshStats getStats() const;

  /** Optimize the table for speed/space
   */
  void optimize();

  friend class LshMatcher;

private:
  /** defines the speed fo the implementation
   * kArray uses a vector for storing data
   * kBitsetHash uses a hash map but checks for the validity of a key with a bitset
   * kHash uses a hash map only
   */
  enum SpeedLevel {kArray, kBitsetHash, kHash};

  /** Compute the key of a feature */
  size_t getKey(const uchar* feature) const;

  /** The vector of all the buckets if they are held for speed */
  BucketsSpeed buckets_speed_;

  /** The vector of all the buckets in case we cannot use the speed version */
  BucketsSpace buckets_space_;

  /** What is used to store the data */
  SpeedLevel speed_level_;

  /** If the subkey is small enough, it will keep track of which subkeys are set through that bitset
   * That is just a speedup so that we don't look in the hash table (which can be mush sloer that checking a bitset)
   */
  boost::dynamic_bitset<> key_bitset_;

  /** The size of the key in bits */
  unsigned int key_size_;

  /** The mask to apply to a feature to get the hash key */
  std::vector<size_t> mask_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class implementing an LSH Matcher for binary features
 * In each cv::Mat the descriptor is a row. The first 8 bits being in the first char, in order 876543210
 * a bit unintuitive, but it simplifies the computations a lot
 */
class LshMatcher : public cv::DescriptorMatcher
{
public:
  LshMatcher() :
    feature_size_(0)
  {
  }

  /** Implementation of the virtual function
   * @param descriptors
   */
  void add(const std::vector<cv::Mat>& descriptors);

  /** Implementation of the virtual function
   */
  void clear();

  /** Implementation of the pure virtual function
   * @return
   */
  inline bool isMaskSupported() const;

  /** Implementation of the virtual function
   */
  void train();

  /** Implementation of the pure virtual function
   * @return
   */
  cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const;

  /** Set certain dimensions in the Matcher
   * @param table_number the number of hash tables to use
   * @param key_size the size of the key
   * @param multi_probe_level how far should we look for neighbors in multi-probe LSH. 0 for standard LSH, 2 is good
   */
  void setDimensions(unsigned int table_number, unsigned int key_size, unsigned int multi_probe_level);

  void getStats(std::vector<LshStats>& stats) const;

private:
  void knnMatchImpl(const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, int k,
                    const std::vector<cv::Mat>& masks = std::vector<cv::Mat>(), bool compactResult = false);
  /** Implementation of the virtual
   * @param queryDescriptors
   * @param matches
   * @param maxDistance
   * @param masks
   * @param compactResult
   */
  void radiusMatchImpl(const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches,
                       float maxDistance, const std::vector<cv::Mat>& masks = std::vector<cv::Mat>(),
                       bool compactResult = false);
  /** Implementation of the virtual
   * @param queryDescriptors
   * @param matches
   * @param maxDistance
   * @param masks
   * @param compactResult
   * @param is_knn
   */
  void match_impl(const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, bool is_knn,
                  float param);

  /** The number of hash tables
   */
  unsigned int table_number_;
  /** The size of the keys (in bits)
   */
  unsigned int key_size_;
  /** Taken from FlannBasedMatcher
   */
  int addedDescCount;

  /** The trainIdx from mergedDescriptors which is protected
   */
  std::vector<int> startIdxs;

  friend class LshTable; //friend for changing the above.
  /** The different hash tables
   */
  std::vector<LshTable> tables_;
  /** Taken from FlannBasedMatcher
   */
  DescriptorCollection mergedDescriptors;
  /** The size of the features (as char[])
   */
  unsigned int feature_size_;

  /** The XOR masks to apply to a key to get the neighboring buckets
   */
  std::vector<BucketKey> xor_masks_;

  /** How far should we look for neighbors in multi-probe LSH
   */
  unsigned int multi_probe_level_;
};
}

#endif /* LSH_HPP_ */
