#include "map_localize/FABMAPLocalizer.h"

#include <time.h>

using namespace cv;

FABMAPLocalizer::FABMAPLocalizer(const std::vector<CameraContainer*>& train, std::string descriptor_type, bool show_matches, bool load, std::string filename)
  : desc_type(descriptor_type), show_matches(show_matches)
{
  keyframes = train;

  detector =  new DynamicAdaptedFeatureDetector(
      AdjusterAdapter::create("SURF"), 1000, 1100);
  extractor = new SurfDescriptorExtractor();//1000, 4, 2, false, true);
  matcher = DescriptorMatcher::create("FlannBased");

  std::string v_filename = filename + std::string("vocab.yml");
  std::string td_filename = filename + std::string("train_data.yml");
  std::string tree_filename = filename + std::string("tree.yml");

  clock_t start_time;
  Mat vocab;
  Mat train_data;
  Mat tree;

  if(load)
  {
    FileStorage fs;
    //load vocab
    std::cout << "Loading Vocabulary: " << v_filename << std::endl;
    fs.open(std::string(v_filename), FileStorage::READ);
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cout << "Vocabulary not found" << std::endl;
        return;
    }
    fs.release();

    bide = new BOWImgDescriptorExtractor(extractor, matcher);
    bide->setVocabulary(vocab);

    std::cout << "Loading Training Data: " << td_filename << std::endl;
    fs.open(td_filename, FileStorage::READ);
    fs["BOWImageDescs"] >> train_data;
    if (train_data.empty()) {
        std::cout << "Training Data not found" << std::endl;
        return;
    }
    fs.release();

    std::cout << "Loading tree: " << tree_filename << std::endl;
    fs.open(tree_filename, FileStorage::READ);
    fs["CLTree"] >> tree;
    if (tree.empty()) {
        std::cout << "Tree not found" << std::endl;
        return;
    }
    fs.release();
    
  }
  else
  {
    std::cout << "Creating vocab..." << std::endl;
    Mat vocab_data;
    for(unsigned int i = 0; i < keyframes.size(); i ++)
    {
      Mat desc;
      vector<KeyPoint> kpts;
      detector->detect(keyframes.at(i)->GetImage(), kpts);
      extractor->compute(keyframes.at(i)->GetImage(), kpts, desc);
      vocab_data.push_back(desc);
      //vocab_data.push_back(keyframes.at(i)->GetDescriptors());
    }

    start_time = clock();
    of2::BOWMSCTrainer bowt;
    vocab = bowt.cluster(vocab_data);
    std::cout << "Vocab created" << std::endl;
    std::cout << double( clock() - start_time ) / (double)CLOCKS_PER_SEC<< " seconds" << std::endl;

    bide = new BOWImgDescriptorExtractor(extractor, matcher);
    bide->setVocabulary(vocab);

    std::cout << "Creating training data..." << std::endl;
    for(unsigned int i = 0; i < keyframes.size(); i++)
    {
      Mat bow;
      vector<KeyPoint> kpts;
      detector->detect(keyframes[i]->GetImage(), kpts);
      bide->compute(keyframes[i]->GetImage(), kpts, bow);
      train_data.push_back(bow);
    }
    std::cout << "Train data created" << std::endl;

    std::cout << "Building tree.." << std::endl;
    start_time = clock();
    of2::ChowLiuTree tree_builder;
    tree_builder.add(train_data);
    tree = tree_builder.make();
    std::cout << "Tree built" << std::endl;
    std::cout << double( clock() - start_time ) / (double)CLOCKS_PER_SEC<< " seconds" << std::endl;

    if(filename != std::string(""))
    {
      std::cout << "Saving vocab and data..." << std::endl;
      FileStorage fs_vocab(v_filename, FileStorage::WRITE);
      if(!fs_vocab.isOpened())
      {
        std::cout << "Failed to save vocab to " << v_filename << std::endl;
        return;
      }
      write(fs_vocab, std::string("Vocabulary"), vocab);
      fs_vocab.release();

      std::cout << "Successfully saved vocab to " << v_filename << std::endl;

      FileStorage fs_train(td_filename, FileStorage::WRITE);
      if(!fs_train.isOpened())
      {
        std::cout << "Failed to save train data to " << td_filename << std::endl;
        return;
      }
      write(fs_train, std::string("BOWImageDescs"), train_data);
      fs_train.release();

      std::cout << "Successfully saved train data to " << td_filename << std::endl;

      FileStorage fs_tree(tree_filename, FileStorage::WRITE);
      if(!fs_tree.isOpened())
      {
        std::cout << "Failed to save tree to " << tree_filename << std::endl;
        return;
      }
      write(fs_tree, std::string("CLTree"), tree);
      fs_tree.release();

      std::cout << "Successfully saved tree to " << tree_filename << std::endl;
    }
  }
  std::cout << "Vocab size = " << vocab.rows << std::endl;


  //fabmap = new of2::FabMap2(tree, 0.39, 0, of2::FabMap::SAMPLED | of2::FabMap::CHOW_LIU);
  //fabmap->addTraining(train_data);
  fabmap = new of2::FabMapLUT(tree, 0.39, 0.05, of2::FabMap::MEAN_FIELD | of2::FabMap::CHOW_LIU);

  fabmap->add(train_data);
}

bool FABMAPLocalizer::localize(Mat& img, Eigen::Matrix4f* pose, Eigen::Matrix4f* pose_guess)
{
  Mat bow;
  vector<KeyPoint> kpts;
  detector->detect(img, kpts);
  bide->compute(img, kpts, bow);

  if(bow.empty())
  {
    std::cout << "Warning: no keypoints detected" << std::endl;
    return false;
  }  

  vector<of2::IMatch> matches;
  fabmap->compare(bow, matches, false);

  if(show_matches)
  {
    //namedWindow( "Query", WINDOW_AUTOSIZE );
    //imshow("Query", img);
    //waitKey(0);

    std::cout << "# comparisons = " << matches.size() << std::endl;
    for(int i = 0; i < matches.size(); i++)
    {
      std::cout << "match prob = " << matches.at(i).match << std::endl;
      std::cout << "match likelihood = " << matches.at(i).likelihood << std::endl;
      std::cout << "imgIdx = " << matches.at(i).imgIdx << " queryIdx = " << matches.at(i).queryIdx << std::endl;
      std::cout << "# keyframes = " << keyframes.size() << std::endl;
      if(matches.at(i).imgIdx > -1 && matches.at(i).match > 0.5)
      {
        namedWindow( "Match", WINDOW_AUTOSIZE );
        imshow("Match", keyframes.at(matches.at(i).imgIdx)->GetImage());
        waitKey(2000);
      }
    }
  }

  for(int i = 0; i < matches.size(); i++)
  {
    if(matches.at(i).imgIdx > -1 && matches.at(i).match > 0.5)
    {
      *pose = keyframes.at(matches.at(i).imgIdx)->GetTf();
      return true;
    }
  }
  return false;
}
