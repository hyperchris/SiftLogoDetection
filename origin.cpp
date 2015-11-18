#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

// pi
#define PI 3.14159265
// sift para
#define MAX_DIST_RATIO 3
#define MINHESSIAN 600
// result dir
#define RES_DIR "res/"
// resize the template into different sizes
#define RESIZE_BASE 60
#define RESIZE_STEP 10
#define RESIZE_NUM 15

/** Get the file name from file path */
string getFileName(string fname) {
  int startPos = fname.find_last_of("/");
  return fname.substr(startPos + 1, fname.length());
}

/** test if the match is good */
bool validResult(Mat & logo, vector<Point2f> & corners) {
  // get the max and min edge of detected area
  double minEdge = 1000;
  double maxEdge = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 4; j++) {
      double edge = norm(corners[i] - corners[j]);
      if (edge > maxEdge) maxEdge = edge;
      if (edge < minEdge) minEdge = edge;
    }
  }
  // compare with logo image (too big or too small, false!)
  if (maxEdge > 3 * logo.cols || minEdge < logo.rows / 2 || minEdge < 15)
    return false;

  // upper and lower edges should be relatively parallel
  // find the upper and lower corners
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3 - i; j++)
      if (corners[j].y < corners[j + 1].y) {
        Point2f temp = corners[j];
        corners[j] = corners[j + 1];
        corners[j + 1] = temp;
      }
  // print result
  for (int i = 0; i < 4; i++)
    cout << corners[i].x << "," << corners[i].y << " ";
  cout << endl;  
  // get two thetas
  double upper_tan = (double)((corners[0] - corners[1]).y) / (double)((corners[0] - corners[1]).x);
  double lower_tan = (double)((corners[2] - corners[3]).y) / (double)((corners[2] - corners[3]).x);
  cout << upper_tan << " " << lower_tan << endl;
  double upper_theta = atan(upper_tan) * 180 / PI;
  double lower_theta = atan(lower_tan) * 180 / PI;
  cout << upper_theta << " " << lower_theta << endl;

  if (abs(upper_theta) > 25 || abs(lower_theta) > 25 || abs(upper_theta - lower_theta) > 30)
    return false;

  return true;
}

/** @function main */
Mat siftMatch(const Mat & img_object, const Mat & img_scene) {
  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = MINHESSIAN;

  SiftFeatureDetector detector(minHessian);

  vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object);
  detector.detect( img_scene, keypoints_scene);

  //-- Step 2: Calculate descriptors (feature vectors)
  SiftDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute(img_object, keypoints_object, descriptors_object);
  extractor.compute(img_scene, keypoints_scene, descriptors_scene);

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  vector<DMatch> matches;
  matcher.match(descriptors_object, descriptors_scene, matches);

  double max_dist = 0; double min_dist = 1000;

  //-- Quick calculation of max and min distances between keypoints
  for(int i = 0; i < descriptors_object.rows; i++) { 
    double dist = matches[i].distance;
    if(dist < min_dist) min_dist = dist;
    if(dist > max_dist) max_dist = dist;
  }

  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  vector< DMatch > good_matches;
  for(int i = 0; i < descriptors_object.rows; i++) { 
    if(matches[i].distance < MAX_DIST_RATIO * min_dist) { 
      good_matches.push_back(matches[i]); 
    }
  }

  //-- Localize the object
  vector<Point2f> obj;
  vector<Point2f> scene;
  for(int i = 0; i < good_matches.size(); i++) {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  return findHomography(obj, scene, CV_RANSAC);
}


Mat logoDetect(const Mat & templ, const string img_path) {
  Mat test_img = imread(img_path, 1);
  Mat img_matches = test_img;
  
  for (int i = 0; i < RESIZE_NUM; i++) {
    // generate different size of template (step = RESIZE_STEP, start from RESIZE_BASE)
    Mat templ_r;
    int templ_r_width = RESIZE_BASE + i * RESIZE_STEP;
    Size size(templ_r_width, templ.rows * templ_r_width / templ.cols);
    resize(templ, templ_r, size);
    // do matching and get H
    Mat H = siftMatch(templ_r, test_img);

    // get the corners and do projection
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint(templ_r.cols, 0);
    obj_corners[2] = cvPoint(templ_r.cols, templ_r.rows); 
    obj_corners[3] = cvPoint(0, templ_r.rows);
    vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);

    // decide if the result is valid and draw result
    if (validResult(templ_r, scene_corners)) {
      line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
      line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
      line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
      line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
    }
  }

  return img_matches;
}

/* --------------------------------------- main --------------------------------------- */
int main(int argc, char** argv) {
  cout << "Input format: [templ_path] [img_path_1] .. [img_path_n] " << endl;
  if (argc < 2) {
    cout << "ERROR: too few arg!" << endl;
    return -1;
  }

  Mat templ = imread(argv[1], 1); // read in the logo template
  printf("processing....");
  for (int i = 2; i < argc; i++) { // for each image
    if (i % 10 == 0) { // show progress
      cout << i << " " << flush;
    }
    imwrite(RES_DIR + getFileName(argv[i]), logoDetect(templ, argv[i]));
  }

  cout << "\ndone" << endl;
  return 0;
}