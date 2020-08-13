

# SFND 3D Object Tracking

This is the final project of the camera course. By completing all the lessons, a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images was achieve. Also, how to detect objects in an image using the YOLO deep-learning framework and know how to associate regions in a camera image with Lidar points in 3D space.

<img src="images/KITTI/course_code_structure.png" width="779" height="414" />

In this final project, was implemented the missing parts in the schematic. To do this, the following four major tasks were completed: 
1. Develop a way to match 3D objects over time by using keypoint correspondences. 
2. Compute the TTC based on Lidar measurements. 
3. Proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. Conduct various tests with the framework, identifying the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.

2. Download dat/yolo/yolov3.weights with Git LFS

   or `! wget "https://pjreddie.com/media/files/yolov3.weights"`

3. Make a build directory in the top level project directory: `mkdir build && cd build`

4. Compile: `cmake .. && make`

5. Run it: `./3D_object_tracking`.

   

## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points

### FP.1 Match 3D Objects

*Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.*

- Solution: Lines 254 ~ 291 at `camFusion_Student.cpp`

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    const int size_p = prevFrame.boundingBoxes.size();
    const int size_c = currFrame.boundingBoxes.size();
  
    cv::Mat count = cv::Mat::zeros(size_p, size_c, CV_32S);
    for (auto matchpair : matches){
        cv::KeyPoint prevkp1 = prevFrame.keypoints.at(matchpair.queryIdx);
        auto prevkp = prevkp1.pt; 
      
        cv::KeyPoint currkp1 = currFrame.keypoints.at(matchpair.trainIdx);
        auto currkp = currkp1.pt;
      
        for (size_t prevbb = 0; prevbb < size_p; prevbb++)
        {
            for (size_t currbb = 0; currbb < size_c; currbb++)
            {
            	if((prevFrame.boundingBoxes[prevbb].roi.contains(prevkp))&&(currFrame.boundingBoxes[currbb].roi.contains(currkp)))
                  count.at<int>(prevbb, currbb) = count.at<int>(prevbb, currbb) + 1;
            }
        }      
    }

    for (size_t i = 0; i < size_p; i++)
    {
        int id = -1;
        int maxvalue = 0;
        for (size_t j = 0; j < size_c; j++)
        {
            if (count.at<int>(i,j) > maxvalue)
            {
                maxvalue = count.at<int>(i,j);
                id = j;
            }
        }
        bbBestMatches[i] = id;
    }                     
}
```

### FP.2 Compute lidar-based TTC

*Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. It return the average on all X points.*

- Solution: Lines 208 ~ 252 at `camFusion_Student.cpp`

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1 / frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane
  	int max_sight = 10;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
  
  	std::vector<double> min_values_xprev, min_values_xcurr;
    double avg_xprev, avg_xcurr;
  
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
      if ((abs(it->y) <= laneWidth/2) && ((it->x)< max_sight)){
          min_values_xprev.push_back(it->x);
          minXPrev = minXPrev > it->x ? it->x : minXPrev;
      }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
      if ((abs(it->y) <= laneWidth/2) && ((it->x)< max_sight)){
          min_values_xcurr.push_back(it->x);
          minXCurr = minXCurr > it->x ? it->x : minXCurr;
      }
    }
  
  	if (min_values_xprev.size() > 0){
    	for(auto point: min_values_xprev)
          	avg_xprev = avg_xprev + point;
      	avg_xprev = avg_xprev / min_values_xprev.size();
    }
    
  	if (min_values_xcurr.size() > 0){
    	for(auto point: min_values_xcurr)
          	avg_xcurr = avg_xcurr + point;
      	avg_xcurr = avg_xcurr / min_values_xcurr.size();
    }
      
    // compute TTC from both measurements
    TTC = avg_xcurr * dT / (avg_xprev - avg_xcurr);
}
```

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

*Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.*

- Solution: Lines 135 ~ 160 at `camFusion_Student.cpp`

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  	double dist = 0;
  	vector<cv::DMatch> tmp_kptMatches;
  
  	for(auto match_pair: kptMatches){
        cv::KeyPoint currkp1 = kptsCurr.at(match_pair.trainIdx);
        auto point = currkp1.pt;

      	if(boundingBox.roi.contains(point)){
          tmp_kptMatches.push_back(match_pair);
          dist += match_pair.distance;
        }
    }
    if (tmp_kptMatches.size() == 0){
      cout << "No keypoints matches where found in this bounding box" << endl;
      return;
    }  
   	dist = dist/tmp_kptMatches.size();
    double threshold = dist * 0.7;
    
  	for(auto pair: tmp_kptMatches){
        if ((pair.distance) < threshold)
          boundingBox.kptMatches.push_back(pair);
    }
    cout << boundingBox.kptMatches.size()  << " keypoints matches added" << endl;
}

```

### FP.4 Compute Camera-based TTC

*Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.*

- Solution: Lines 166 ~ 206 at `camFusion_Student.cpp`

```c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; 

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { 
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }     

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

}
```

## FP.5 Performance Evaluation 1

*Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.*

2 inconsistencies were found, in frames 3 and 17.  During frame 2 to 3, the distance between the car and the preceding car does not change or even decreases, while the TTC predict from Lidar increases significantly (from 13s to 16s). Looking at the top view image generated from the frame, it is possible to notice some outliers being counted. The same happens in frame 16 to 17, where the car is clearly getting closer while the TCC jumps from 8s to 11s. Once again, a couple of outliers are spotted.

- Frame 3

![Screenshot from 2020-07-20 11-21-02](/home/jacques/Projects/Sensor Fusion /Camera/Screenshot from 2020-07-20 11-21-02.png)



![Screenshot from 2020-07-20 11-23-26](/home/jacques/Projects/Sensor Fusion /Camera/Screenshot from 2020-07-20 11-23-26.png)



- Frame 17

![Screenshot from 2020-07-20 11-21-17](/home/jacques/Projects/Sensor Fusion /Camera/Screenshot from 2020-07-20 11-21-17.png)

![Screenshot from 2020-07-20 11-23-43](/home/jacques/Projects/Sensor Fusion /Camera/Screenshot from 2020-07-20 11-23-43.png)



### FP.6 Performance Evaluation 2

*Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.*

A loop was create and all the results were saved in a 2D Array to generate a table, called FP_6_Performance_Evaluation_Matrix.csv, saved in the above directory. Analysing the CSV file through a XLSX file, the following results were achieve.

```c++
  	// Matheus var
  	bool INFO = false;  
    std::vector<std::string> detector_type_names = { "SHITOMASI", "FAST", "BRISK", "ORB", "AKAZE"};
    std::vector<std::string> descriptor_type_names = {"BRISK", "BRIEF", "ORB", "FREAK"};
  	int n_col = detector_type_names.size()*descriptor_type_names.size();
  	double perfomance_eval[imgEndIndex][n_col];
  	int tracker = 0;
  
    for(auto detector_type_name:detector_type_names) // start loop detector_types and descriptor_types
    {...}
    
    ofstream performance_eval_csv;
  
    performance_eval_csv.open("../FP_6_Performance_Evaluation_Matrix.csv", std::ios::out);
	  
  	for(auto detector_type_name:detector_type_names) // Fill csv with Det/Des pairs
    {...} 
    performance_eval_csv << endl;
  
    for(int i=0; i< imgEndIndex; i++) // Write perfomance_eval array to CSV 
    {...}
    performance_eval_csv.close( );
  
    return 0;
```
Here are the TCC's for Lidar estimation:

| Frame | TCC Lidar |
| ----- | --------- |
| 1     | 12.2891   |
| 2     | 13.3547   |
| 3     | 16.3845   |
| 4     | 14.0764   |
| 5     | 12.7299   |
| 6     | 13.7511   |
| 7     | 13.7314   |
| 8     | 13.7901   |
| 9     | 12.059    |
| 10    | 11.8642   |
| 11    | 11.9682   |
| 12    | 9.8871    |
| 13    | 9.4250    |
| 14    | 9.3021    |
| 15    | 8.3212    |
| 16    | 8.8986    |
| 17    | 11.0301   |
| 18    | 8.5355    |

and this file (FP_6_Performance_Evaluation_Matrix.xlsx) we see TCC's based in the pair Detector/Descriptor. The TOP 3 pairs are:

| **Detector/Descriptor** | Avg Error when compared to Lidar's TCC |
| ----------------------- | -------------------------------------- |
| AKAZE_FREAK             | -0,003962222222                        |
| SHITOMASI_FREAK         | 0,06118777778                          |
| AKAZE_BRISK             | 0,07296444444                          |

The ORB Detector had the worst performance.
