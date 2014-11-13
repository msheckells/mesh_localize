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

#include <boost/foreach.hpp>
#include <boost/functional/hash.hpp>
#include <iostream>
#include <iomanip>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "map_localize/lsh.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
void fill_xor_mask(lsh::BucketKey key, int lowest_index, unsigned int level, std::vector<lsh::BucketKey> & xor_masks)
{
  xor_masks.push_back(key);
  if (level == 0)
    return;
  for (int index = lowest_index - 1; index >= 0; --index)
  {
    // Create a new key
    lsh::BucketKey new_key = key | (1 << index);
    fill_xor_mask(new_key, index, level - 1, xor_masks);
  }
}
}

namespace lsh
{
/** Constructor
 * Create the mask and allocate the memory
 * @param feature_size is the size of the feature (considered as a char[])
 * @param key_size is the number of bits that are turned on in the feature
 */
LshTable::LshTable(unsigned int feature_size, unsigned int key_size) :
  speed_level_(kHash), key_size_(key_size)
{
  // Allocate the mask
  mask_ = std::vector<size_t>(ceil((float)(feature_size * sizeof(char)) / (float)sizeof(size_t)), 0);

  // Generate a random set of order of key_size_ bits
  boost::dynamic_bitset<> allocated_bits(feature_size * CHAR_BIT);
  while (allocated_bits.count() < key_size_)
  {
    size_t index = std::rand() % allocated_bits.size();
    if (allocated_bits.test(index))
      continue;
    allocated_bits.set(index);

    // Set that bit in the mask
    size_t divisor = CHAR_BIT * sizeof(size_t);
    size_t idx = index / divisor; //pick the right size_t index
    mask_[idx] += size_t(1) << (index % divisor); //use modulo to find the bit offset
  }

#define PRINT_MASK_DEBUG 0
#if PRINT_MASK_DEBUG
  {
    printMask(std::cout);
  }
#endif
}

/** Get a bucket given the key
 * @param key
 * @return
 */
inline const Bucket * LshTable::getBucketFromKey(BucketKey key) const
{
  // Generate other buckets
  switch (speed_level_)
  {
    case kArray:
      // That means we get the buckets from an array
      return &buckets_speed_[key];
      break;
    case kBitsetHash:
      // That means we can check the bitset for the presence of a key
      if (key_bitset_.test(key))
        return &buckets_space_.at(key);
      else
        return 0;
      break;
    case kHash:
    {
      // That means we have to check for the hash table for the presence of a key
      BucketsSpace::const_iterator bucket_it, bucket_end = buckets_space_.end();
      bucket_it = buckets_space_.find(key);
      // Stop here if that bucket does not exist
      if (bucket_it == bucket_end)
        return 0;
      else
        return &bucket_it->second;
      break;
    }
  }
  return 0;
}

std::ostream& LshTable::printMask(std::ostream& out) const
{
  size_t bcount = 0;
  BOOST_FOREACH(size_t mask_block, mask_)
        {
          out << std::setw(sizeof(size_t) * CHAR_BIT / 4) << std::setfill('0') << std::hex << mask_block << std::endl;
          bcount += __builtin_popcountll(mask_block);
        }
  out << "bit count : " << std::dec << bcount << std::endl;
  out << "mask size : " << mask_.size() << std::endl;
  return out;
}

LshStats LshTable::getStats() const
{
  LshStats stats;
  stats.bucket_size_mean_ = 0;
  if ((buckets_speed_.empty()) && (buckets_space_.empty()))
  {
    stats.n_buckets_ = 0;
    stats.bucket_size_median_ = 0;
    stats.bucket_size_min_ = 0;
    stats.bucket_size_max_ = 0;
    return stats;
  }

  if (!buckets_speed_.empty())
  {
    BOOST_FOREACH(const Bucket& bucket, buckets_speed_)
          {
            stats.bucket_sizes_.push_back(bucket.size());
            stats.bucket_size_mean_ += bucket.size();
          }
    stats.bucket_size_mean_ /= buckets_speed_.size();
    stats.n_buckets_ = buckets_speed_.size();
  }
  else
  {
    BOOST_FOREACH(const BucketsSpace::value_type& x, buckets_space_)
          {
            stats.bucket_sizes_.push_back(x.second.size());
            stats.bucket_size_mean_ += x.second.size();
          }
    stats.bucket_size_mean_ /= buckets_space_.size();
    stats.n_buckets_ = buckets_space_.size();
  }

  std::sort(stats.bucket_sizes_.begin(), stats.bucket_sizes_.end());

  //  BOOST_FOREACH(int size, stats.bucket_sizes_)
  //          std::cout << size << " ";
  //  std::cout << std::endl;
  stats.bucket_size_median_ = stats.bucket_sizes_[stats.bucket_sizes_.size() / 2];
  stats.bucket_size_min_ = stats.bucket_sizes_.front();
  stats.bucket_size_max_ = stats.bucket_sizes_.back();

  cv::Scalar mean, stddev;
  cv::meanStdDev(cv::Mat(stats.bucket_sizes_), mean, stddev);
  stats.bucket_size_std_dev = stddev[0];

  // Include a histogram of the buckets
  unsigned int bin_start = 0;
  unsigned int bin_end = 20;
  bool is_new_bin = true;
  for (std::vector<unsigned int>::iterator iterator = stats.bucket_sizes_.begin(), end = stats.bucket_sizes_.end(); iterator
      != end;)
    if (*iterator < bin_end)
    {
      if (is_new_bin)
      {
        stats.size_histogram_.push_back(std::vector<unsigned int>(3, 0));
        stats.size_histogram_.back()[0] = bin_start;
        stats.size_histogram_.back()[1] = bin_end - 1;
        is_new_bin = false;
      }
      ++stats.size_histogram_.back()[2];
      ++iterator;
    }
    else
    {
      bin_start += 20;
      bin_end += 20;
      is_new_bin = true;
    }

  return stats;
}

std::ostream& operator <<(std::ostream& out, const LshStats & stats)
{
  size_t w = 20;
  out << "Lsh Table Stats:\n" << std::setw(w) << std::setiosflags(std::ios::right) << "N buckets : "
      << stats.n_buckets_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "mean size : "
      << std::setiosflags(std::ios::left) << stats.bucket_size_mean_ << "\n" << std::setw(w)
      << std::setiosflags(std::ios::right) << "median size : " << stats.bucket_size_median_ << "\n" << std::setw(w)
      << std::setiosflags(std::ios::right) << "min size : " << std::setiosflags(std::ios::left)
      << stats.bucket_size_min_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "max size : "
      << std::setiosflags(std::ios::left) << stats.bucket_size_max_;

  // Display the histogram
  out << std::endl << std::setw(w) << std::setiosflags(std::ios::right) << "histogram : "
      << std::setiosflags(std::ios::left);
  for (std::vector<std::vector<unsigned int> >::const_iterator iterator = stats.size_histogram_.begin(), end =
      stats.size_histogram_.end(); iterator != end; ++iterator)
    out << (*iterator)[0] << "-" << (*iterator)[1] << ": " << (*iterator)[2] << ",  ";

  return out;
}

/** Add a feature to the table
 * @param value the value to store for that feature
 * @param feature the feature itself
 */
void LshTable::add(unsigned int value, const cv::Mat & feature)
{
  // Add the value to the corresponding bucket
  BucketKey key = getKey(feature.data);

  switch (speed_level_)
  {
    case kArray:
      // That means we get the buckets from an array
      buckets_speed_[key].push_back(value);
      break;
    case kBitsetHash:
      // That means we can check the bitset for the presence of a key
      key_bitset_.set(key);
      buckets_space_[key].push_back(value);
      break;
    case kHash:
    {
      // That means we have to check for the hash table for the presence of a key
      buckets_space_[key].push_back(value);
      break;
    }
  }
}

/** Return the key of a feature
 * @param feature the feature to analyze
 */
size_t LshTable::getKey(const uchar* feature) const
{
  // no need to check if T is dividable by sizeof(size_t) like in the Hamming
  // distance computation as we have a mask
  const size_t *feature_block_ptr = reinterpret_cast<const size_t*> (feature);

  // Figure out the key of the feature
  // Given the feature ABCDEF, and the mask 001011, the output will be
  // 000CEF
  size_t key = 0;
  size_t bit_index = 1;

  BOOST_FOREACH(size_t mask_block, mask_)
        {
          // get the mask and signature blocks
          size_t feature_block = *feature_block_ptr;
          while (mask_block)
          {
            // Get the lowest set bit in the mask block
            size_t lowest_bit = mask_block & (-mask_block);
            // Add it to the current key if necessary
            key += (feature_block & lowest_bit) ? bit_index : 0;
            // Reset the bit in the mask block
            mask_block ^= lowest_bit;
            // increment the bit index for the key
            bit_index <<= 1;
          }
          // Check the next feature block
          ++feature_block_ptr;
        }
  return key;
}

/** Optimize the table for speed/space
 */
void LshTable::optimize()
{
  // If we are already using the fast storage, no need to do anything
  if (speed_level_==kArray)
    return;

  // Use an array if it will be more than half full
  if (buckets_space_.size() > (unsigned int)((1 << key_size_) / 2)) {
    speed_level_ = kArray;
    // Fill the array version of it
    buckets_speed_.resize(1 << key_size_);
    BOOST_FOREACH(const BucketsSpace::value_type & key_bucket, buckets_space_)
            buckets_speed_[key_bucket.first] = key_bucket.second;

    // Empty the hash table
    buckets_space_.clear();
    return;
  }

  // If the bitset is going to use less than 10% of the RAM of the hash map (at least 1 size_t for the key and two
  // for the vector) or less than 512MB (key_size_ <= 30)
  if (((std::max(buckets_space_.size(), buckets_speed_.size()) * CHAR_BIT * 3 * sizeof(BucketKey)) / 10
      >= size_t(1 << key_size_)) || (key_size_ <= 32))
  {
    speed_level_ = kBitsetHash;
    key_bitset_.resize(1 << key_size_);
    key_bitset_.reset();
    // Try with the BucketsSpace
    BOOST_FOREACH(const BucketsSpace::value_type & key_bucket, buckets_space_)
            key_bitset_.set(key_bucket.first);
  } else {
    speed_level_ = kHash;
    key_bitset_.clear();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Implementation of the virtual function
 * @param descriptors
 */
void LshMatcher::add(const std::vector<cv::Mat>& descriptors)
{
  if ((feature_size_ == 0) && (!descriptors.empty()) && (!descriptors[0].empty()))
  {
    switch (descriptors[0].depth())
    {
      case CV_8U:
      case CV_8S:
        feature_size_ = 8;
        break;
      case CV_16U:
      case CV_16S:
        feature_size_ = 16;
        break;
      case CV_32S:
      case CV_32F:
        feature_size_ = 32;
        break;
      case CV_64F:
        feature_size_ = 64;
        break;
    };
    feature_size_ = descriptors[0].cols * feature_size_ / CHAR_BIT;
    tables_.clear();
    // Compute the distribution of each bit
    cv::Mat_<float> prior(1, feature_size_ * CHAR_BIT);

    for (unsigned int i = 0; i < table_number_; ++i)
      tables_.push_back(LshTable(feature_size_, key_size_));
  }
  // Taken from FlannBasedMatcher
  DescriptorMatcher::add(descriptors);
  for (size_t i = 0; i < descriptors.size(); i++)
    addedDescCount += descriptors[i].rows;
}

/** Implementation of the virtual function
 */
void LshMatcher::clear()
{
  // Taken from FlannBasedMatcher
  DescriptorMatcher::clear();

  mergedDescriptors.clear();

  addedDescCount = 0;

  // Proper to LSH
  tables_.clear();
  table_number_ = 0;
  key_size_ = 0;
}

/** Implementation of the pure virtual function
 * @return
 */
cv::Ptr<cv::DescriptorMatcher> LshMatcher::clone(bool emptyTrainData) const
{
  LshMatcher* matcher = new LshMatcher(*this);
  if (!emptyTrainData)
  {
    matcher->addedDescCount = addedDescCount;
    matcher->mergedDescriptors = DescriptorCollection(mergedDescriptors);
    matcher->tables_ = tables_;
    matcher->feature_size_ = feature_size_;
  }
  matcher->setDimensions(table_number_, key_size_, 2);
  return matcher;
}

void LshMatcher::getStats(std::vector<LshStats>& stats) const
{
  BOOST_FOREACH(const LshTable&table, tables_)
          stats.push_back(table.getStats());
}

/** Implementation the pure virtual function
 * @return
 */
bool LshMatcher::isMaskSupported() const
{
  return true;
}

/** Set certain dimensions in the Matcher
 * @param feature_size the size of thefeature as a char[]
 * @param table_number the number of hash tables to use
 * @param key_size the size of the key
 * @param multi_probe_level how far should we look for neighbors in multi-probe LSH. 0 for standard LSH, 2 is good
 */
void LshMatcher::setDimensions(unsigned int table_number, unsigned int key_size, unsigned int multi_probe_level)
{
  clear();
  table_number_ = table_number;
  key_size_ = key_size;
  multi_probe_level_ = multi_probe_level;

  fill_xor_mask(0, key_size_, multi_probe_level_, xor_masks_);

  // Re-add all the descriptors that we have
  add(getTrainDescriptors());
}

/** Implementation of the virtual function
 */
void LshMatcher::train()
{
  int previousSize = mergedDescriptors.size();
  if (previousSize < addedDescCount)
  {
    mergedDescriptors.set(trainDescCollection);
    // update startIdxs
    startIdxs.resize(trainDescCollection.size());
    startIdxs[0] = 0;
    for (unsigned int i = 1; i < trainDescCollection.size(); ++i)
      startIdxs[i] = startIdxs[i - 1] + trainDescCollection[i - 1].rows;
    // Add the missing entries to the different tables
    BOOST_FOREACH(LshTable & table, tables_)
          {
            // TODO add preallocation for a hash map, not necessary with boost though, maybe with c+++0x
            for (int i = previousSize; i < addedDescCount; ++i)
              table.add(i, mergedDescriptors.getDescriptor(i));
            // Now that the table is full, optimize it for speed/space
            table.optimize();
          }
  }
}

void LshMatcher::knnMatchImpl(const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, int k,
                              const std::vector<cv::Mat>& masks, bool compactResult)
{
  match_impl(queryDescriptors, matches, true, k);
}

/** Implementation of the virtual
 * @param queryDescriptors
 * @param matches
 * @param maxDistance
 * @param masks
 * @param compactResult
 */
void LshMatcher::radiusMatchImpl(const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches,
                                 float maxDistance, const std::vector<cv::Mat>& masks, bool compactResult)
{
  match_impl(queryDescriptors, matches, false, maxDistance);
}

/** Defines the comparator on score and index
 */
struct ScoreIndex
{
public:
  ScoreIndex(unsigned int distance, unsigned int index) :
    distance_(distance), index_(index)
  {
  }
  bool operator==(const ScoreIndex & score_index)
  {
    return ((index_ == score_index.index_) && (distance_ == score_index.distance_));
  }
  bool operator<(const ScoreIndex & score_index)
  {
    return ((distance_ < score_index.distance_) || ((distance_ == score_index.distance_) && (index_
        < score_index.index_)));
  }
  unsigned int distance_;
  unsigned int index_;
};
struct SortScoreIndexPairOnIndex
{
  bool operator()(const ScoreIndex &left, const ScoreIndex &right) const
  {
    return left.index_ < right.index_;
  }
};

struct DMatchComparator
{
  bool operator()(const cv::DMatch &left, const cv::DMatch &right) const
  {
    return left.distance < right.distance;
  }
};

/** Implementation of the virtual
 * @param queryDescriptors
 * @param matches
 * @param maxDistance
 * @param masks
 * @param compactResult
 * @param is_knn
 */
void LshMatcher::match_impl(const cv::Mat& query_descriptors, std::vector<std::vector<cv::DMatch> >& matches,
                            bool is_knn, float param)
{
  cv::Hamming hammingDistanceOperator;
  matches.clear();
  matches.resize(query_descriptors.rows);

  std::vector<ScoreIndex> score_index_heap;
  if (is_knn)
    score_index_heap.reserve(param);

  //set the static checked_average
  const cv::Mat & descriptors = mergedDescriptors.getDescriptors();

  for (int queryIndex = 0; queryIndex < query_descriptors.rows; ++queryIndex)
  {
    const cv::Mat& current_descriptor = query_descriptors.row(queryIndex);

    // Figure out a list of unique indices to query
    score_index_heap.clear();

    // Go over each descriptor index
    unsigned int hamming_distance;
    if (is_knn)
    {
      unsigned int worst_score = std::numeric_limits<unsigned int>::max();
      BOOST_FOREACH(const LshTable&table, tables_)
            {
              size_t key = table.getKey(current_descriptor.data);
              BOOST_FOREACH(BucketKey xor_mask, xor_masks_)
                    {
                      size_t sub_key = key ^ xor_mask;
                      const Bucket * bucket = table.getBucketFromKey(sub_key);
                      if (bucket == 0)
                        continue;

                      BOOST_FOREACH(FeatureIndex training_index, *bucket)
                            {
                              // Compute the Hamming distance
                              hamming_distance = hammingDistanceOperator(
                                                                         current_descriptor.data,
                                                                         descriptors.data + training_index
                                                                             * descriptors.step,
                                                                         current_descriptor.cols);

                              if (hamming_distance < worst_score)
                              {
                                // Figure out where to insert the new element
                                std::vector<ScoreIndex>::iterator score_index = score_index_heap.end() - 1,
                                                                  score_index_end = score_index_heap.begin() - 1;
                                ScoreIndex new_score_index(hamming_distance, training_index);
                                for (; score_index != score_index_end; --score_index)
                                {
                                  if (new_score_index < *score_index)
                                    continue;
                                  else
                                    break;
                                }
                                ++score_index;
                                // Make sur eit is a unique value
                                if (new_score_index == *score_index)
                                  continue;
                                // Insert it
                                score_index_heap.insert(score_index, new_score_index);
                                if (score_index_heap.size() > (unsigned int)param)
                                {
                                  // Remove the highest distance value as we have too many elements
                                  score_index_heap.pop_back();

                                  // Keep track of the worst score
                                  worst_score = score_index_heap.back().distance_;
                                }
                              }

                            }
                    }
            }
    }
    else
    {
      BOOST_FOREACH(const LshTable&table, tables_)
            {
              size_t key = table.getKey(current_descriptor.data);
              BOOST_FOREACH(BucketKey xor_mask, xor_masks_)
                    {
                      size_t sub_key = key ^ xor_mask;
                      const Bucket * bucket = table.getBucketFromKey(sub_key);
                      if (bucket == 0)
                        continue;

                      BOOST_FOREACH(FeatureIndex training_index, *bucket)
                            {
                              // Compute the Hamming distance
                              hamming_distance = hammingDistanceOperator(
                                                                         current_descriptor.data,
                                                                         descriptors.data + training_index
                                                                             * descriptors.step,
                                                                         current_descriptor.cols);
                              if (hamming_distance < param)
                                score_index_heap.push_back(ScoreIndex(hamming_distance, training_index));
                            }
                    }
            }
    }

    // Store the matches
    std::vector<int>::const_iterator first_image_iterator = startIdxs.begin(), last_image_iterator = startIdxs.end();
    std::sort(score_index_heap.begin(), score_index_heap.end(), SortScoreIndexPairOnIndex());
    BOOST_FOREACH(const ScoreIndex &score_index, score_index_heap)
          {
            int imgIdx, localDescIdx, globalDescIdx = score_index.index_;

            // Figure out the local index of the global index (faster than calling getLocalIdx several times) as the
            // indices are already sorted
            first_image_iterator = std::upper_bound(first_image_iterator, last_image_iterator, globalDescIdx);
            --first_image_iterator;
            imgIdx = first_image_iterator - startIdxs.begin();
            localDescIdx = globalDescIdx - (*first_image_iterator);

            matches[queryIndex].push_back(cv::DMatch(queryIndex, localDescIdx, imgIdx, score_index.distance_));
          }
    // Remember to sort the matches with respect to their score
    std::sort(matches[queryIndex].begin(), matches[queryIndex].end(), DMatchComparator());
  }
}
}
