#include <iostream> // for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

struct MouseParams
{
    Mat img;
    Mat ori;
    Point2f pt;
};

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		Mat* img = (Mat*)userdata;  // 1st cast it back, then deref
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		cout << img->at<Vec3b>( Point(x, y) ) << endl;
	}
}

void BallSelectFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		MouseParams* param = (MouseParams*)userdata;  // 1st cast it back, then deref
		// reset img & draw circle
		param->ori.copyTo( param->img );
		line(param->img, Point(x-5, y), Point(x+5, y), CV_RGB(255,0,0), 2);	
		line(param->img, Point(x, y-5), Point(x, y+5), CV_RGB(255,0,0), 2);	
        // update ball_pick
        param->pt = Point2f(x, y);
		cout << "Ball starting point - position (" << x << ", " << y << ")" << endl;
	}
}

// Basketball Color 
int iLowH = 180;
int iHighH = 16;

int iLowS =  95;
int iHighS = 200;

int iLowV = 75;
int iHighV = 140;

/*
int iLowH = 0;
int iHighH = 20;

int iLowS = 50;
int iHighS = 255;

int iLowV = 50;
int iHighV = 160;
*/

Mat getMask(Mat &frameHSV)
{
	Mat mask1, mask2, mask;
	inRange(frameHSV, Scalar(0, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);
	inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2);

	mask = mask1 + mask2;

	// morphological opening (remove small objects from the foreground)
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );

	// morphological closing (fill small holes in the foreground)
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
	
	return mask;
}

int main(int argc, char** argv)
{
	if(argc >= 3)
	{
		VideoCapture inputVideo(argv[1]); // open the default camera
		if(!inputVideo.isOpened())  // check if we succeeded
		    return -1; 
		
		// Initialize
	    VideoWriter outputVideo;  // Open the output
	    const string source      = argv[2];                                // the source file name
		const string NAME = source + ".mp4";   // Form the new name with container
	    int ex = inputVideo.get(CV_CAP_PROP_FOURCC);                       // Get Codec Type- Int form
		std::cout << ex << "\n" << (int)inputVideo.get(CV_CAP_PROP_FOURCC) << "\n";
    	Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),       //Acquire input size
        	          (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));    
		outputVideo.open(NAME, ex, inputVideo.get(CV_CAP_PROP_FPS), S, false);
    	char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
	    cout << "Input codec type: " << EXT << endl;

        if (!outputVideo.isOpened())
        {
            cout  << "Could not open the output video for write \n";
            return -1;
        }
        
		namedWindow("Result Window", 1);
		
		// Mat declaration
		Mat prev_frame, prev_gray, cur_frame, cur_gray;
        Mat frame_blurred, frameHSV;
        
        // take the first frame
        inputVideo >> prev_frame;
		
        /* manual ball selection */
        MouseParams mp;
        prev_frame.copyTo( mp.ori ); 
        prev_frame.copyTo( mp.img ); 
        setMouseCallback("Result Window", BallSelectFunc, &mp );

		int enterkey = 0;
		while(enterkey != 32 && enterkey != 113)
		{
			enterkey = waitKey(30) & 0xFF;
        	imshow("Result Window", mp.img);
		}
		
		/* Kalman Filter
		   Kalman filter is a prediction-correction filter. It has two stages: predict and correct.
		   In predict stage, the filter uses the states of previous frame to predict the
		   state of current frame. In correct stage, the filter takes in current measurement
		   to "correct" the prediction made in prediction stage. 
		   Here we are using an adaptive Kalman filter to do ball tracking.
		   (noise matrix P, Q changes depending on the occulusion index)
		*/
		
		/* Initialization
		   four parameters:  x, y, vx, vy
		   two measurements: mx, my
		   Here we're implementing a constant velocity model.
		   x_t = x_t-1 + vx_t-1;
		   y_t = y_t-1 + vy_t-1;
		   vx_t = vx_t-1;
		   vy_t = vy_t-1;
		   These linear equations can be written as transition matrix A.
		*/
		KalmanFilter KF(4, 2, 0);
		float transMatrixData[16] = {1,0,1,0, 
		                             0,1,0,1,
		                             0,0,1,0,
		                             0,0,0,1};
		                             
		KF.transitionMatrix = Mat(4, 4, CV_32F, transMatrixData);
		Mat_<float> measurement(2,1);
		measurement.setTo(Scalar(0));
		
		/* We put the first point in predicted state */
		KF.statePost.at<float>(0) = mp.pt.x;
		KF.statePost.at<float>(1) = mp.pt.y;
		KF.statePost.at<float>(2) = 0;
		KF.statePost.at<float>(3) = 0;
		setIdentity(KF.measurementMatrix);                        // measurement matrix H
		setIdentity(KF.processNoiseCov, Scalar::all(1e-4));       // process noise covariance matrix Q
		setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));   // measurement noise covariance matrix R
		// priori error estimate covariance matrix P'(t)		
		/*
		KF.errorCovPre.at<float>(0)  = 1;
		KF.errorCovPre.at<float>(5)  = 1;
		KF.errorCovPre.at<float>(10) = 1;
		KF.errorCovPre.at<float>(15) = 1;   
		*/
		setIdentity(KF.errorCovPre);                              // priori error estimate covariance matrix P'(t)	
		setIdentity(KF.errorCovPost, Scalar::all(.1));            // posteriori error estimate cov matrix P(t)
	
		/* params related to previous frames */
		Rect    prev_box;
		Point2f prev_motion;
		Point   noFoundStartPt;
        vector<cv::Point2f> prev_ball_centers;
        int noFoundCount = 0;
        
        /* start tracking */		
		setMouseCallback("Result Window", CallBackFunc, &frameHSV);			
        for(int frame_num=1; frame_num < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_num)
        {
        	cout << "===FRAME #" << frame_num << "===" << endl;
        	
        	/* get current frame */
            inputVideo >> cur_frame;
            
            // Blur & convert frame to HSV color space
            cv::GaussianBlur(cur_frame, frame_blurred, cv::Size(5, 5), 3.0, 3.0);
            cvtColor(frame_blurred, frameHSV, COLOR_BGR2HSV);
            
            // gray scale current frame
    		cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);            
    		cvtColor(cur_frame, cur_gray, CV_BGR2GRAY);
            
            // mask generation
            Mat mask;
			mask = getMask(frameHSV);

			// Hough Transform
			Mat frame_filtered, frame_filtered_gray;
            cur_frame.copyTo( frame_filtered, mask );
            cv::cvtColor( frame_filtered, frame_filtered_gray, CV_BGR2GRAY );
            vector<cv::Vec3f> circles;
            cv::GaussianBlur(frame_filtered_gray, frame_filtered_gray, cv::Size(5, 5), 3.0, 3.0);
            HoughCircles( frame_filtered_gray, circles, CV_HOUGH_GRADIENT, 1, frame_filtered_gray.rows/8, 120, 18, 5,300);
			
            // contour generation
            vector< vector<cv::Point> > contours_ball;
            cv::findContours(mask, contours_ball, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            
            Mat result;
            cur_frame.copyTo( result );

			// OpticalFlow
            vector<Point2f> optFlow_ball_centers;
            vector<uchar> featuresFound;
            Mat err;
            TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
            Size winSize(31, 31);
            if( prev_ball_centers.size() > 0 )
                calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_ball_centers, optFlow_ball_centers, featuresFound, err, winSize, 0, termcrit, 0, 0.001);    
            
            // Kalman Filter: Extract previous point & prediction point
            Point2f statePt   = Point( KF.statePost.at<float>(0), KF.statePost.at<float>(1) );  
            Mat prediction    = KF.predict();  
            Point2f predictPt = Point2f( prediction.at<float>(0), prediction.at<float>(1) );

		    cout << "state:" << statePt << endl;
		    cout << "predict:" << predictPt << endl;
			cout << "prev_motion: " << prev_motion << " sqr: " << prev_motion.x * prev_motion.x + prev_motion.y * prev_motion.y << endl;
            
            // Search current frame for good candidate measurements
            vector<Point2f>   cur_contour_centers;
            vector<cv::Point> best_ball_contour;
            Point2f best_ball_center;
            Rect    best_ball_box;
			bool 	ballFound = false;
			
			// TODO dynamic search range
			int closest_dist = (prev_motion.x * prev_motion.x + prev_motion.y * prev_motion.y) * 16;
	    	if(closest_dist == 0) closest_dist = 10000;
	    	// circle( result, predictPt, sqrt(closest_dist), CV_RGB(255,255,0), 2 );
			
            for (size_t i = 0; i < contours_ball.size(); i++)
			{  			    
			    drawContours(result, contours_ball, i, CV_RGB(255,0,0), 1);  // draw the area
			     
		        cv::Rect bBox;
		        bBox = cv::boundingRect(contours_ball[i]);
			    Point2f center;
			    center.x = bBox.x + bBox.width / 2;
			    center.y = bBox.y + bBox.height / 2;		         

				cur_contour_centers.push_back(center);
				
				// find corresponding optical flow center
				float optFlow_dist = 2500;
				int   best_j = -1;
				for( size_t j=0; j < optFlow_ball_centers.size(); ++j )
				{
		        	float diff_x = center.x - optFlow_ball_centers[j].x;
		        	float diff_y = center.y - optFlow_ball_centers[j].y;
		        	float distance  = diff_x * diff_x + diff_y * diff_y;
					if(distance < optFlow_dist)
					{
						distance = optFlow_dist;
						best_j   = j;
					}
			    }
			    
				/* TODO
				Point2f optPredictPt = center;
				if(best_j != -1)
				{
					Point2f motion = optFlow_ball_centers[best_j] - prev_ball_centers[best_j];
					optPredictPt = center + motion;
					line( result, optPredictPt, center, CV_RGB(190,60,70), 2 );
				}
				*/
					
		        // If we find a contour that includes our prediction point,
		        // it's the best choice then.
				// If we cannot found a contour to contain prediction point, 
				// we search the rest contours. The one with closest distance
				// should be picked.
				if( pointPolygonTest( contours_ball[i], predictPt, false ) >= 0)
				{
					best_ball_contour = contours_ball[i];
					best_ball_center  = center;
					best_ball_box     = bBox;
					ballFound = true;
					break;
				}
				else 
				{
		        	float diff_x = center.x - predictPt.x;
		        	float diff_y = center.y - predictPt.y;
		        	float distance  = diff_x * diff_x + diff_y * diff_y;
					
					//if( bBox.area() < 200 )
					//	continue;
					/*
					stringstream sstr;
					sstr << "(dot= " << dot_product << ")";
					cv::putText(result, sstr.str(),
					cv::Point(center.x + 3, center.y - 3),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,255,100), 2);						
					*/
					
					// if distance is close enough
					if( distance < closest_dist )
					{
						best_ball_contour = contours_ball[i];
						best_ball_center  = center;
						best_ball_box     = bBox;		
						closest_dist      = distance;
						ballFound = true;
					}
				}
            }
	
			if(ballFound)
	        {
	        	// calculte occulusion rate
			    float occ = fabs( (float)best_ball_box.area() / (float)prev_box.area() - 1.0 );
			    if(occ > 1.0) occ = 1.0;
				
				// check threshold
				float threshold = 0.3;
				if(occ < threshold)
				{				
					setIdentity(KF.processNoiseCov, Scalar::all(1.0-occ));  // Q = 1 - occ
					setIdentity(KF.measurementNoiseCov, Scalar::all(occ));  // R = occ    
				}
				else
				{
					setIdentity(KF.processNoiseCov, Scalar::all(0.0));      // Q = 0
					setIdentity(KF.measurementNoiseCov, Scalar::all(1e10)); // R = infinite			    				
					cout << "NON_CONFIDENTIAL_MEASUREMENT\n";
				}
				
				// correction
			    measurement.at<float>(0) = best_ball_center.x;  
				measurement.at<float>(1) = best_ball_center.y;  
				Mat estimated = KF.correct(measurement);
			
				cout << "measured:" << best_ball_center << endl;
				cout << "estimated:" << estimated.at<float>(0) << ", " << estimated.at<float>(1) << endl;
          	
				// remember to update prev parameters
				prev_box     = best_ball_box;
				prev_motion  = best_ball_center - statePt;
				noFoundCount = 0;
		    } 
		    else
		    {
				// TODO
				prev_motion = predictPt - statePt;
				
				if( noFoundCount == 0 )
				{
					noFoundStartPt = statePt;
				}
    		    circle( result, noFoundStartPt, 5, CV_RGB(255,255,255), 2 );
				
				// if Kalman filter failed... we "CORRECT" the frame
				if(noFoundCount > 5)
				{
					closest_dist = 1e8;
				    for( size_t i = 0; i < contours_ball.size(); i++ )
				    {                
						cv::Rect bBox;
						bBox = cv::boundingRect(contours_ball[i]);
						Point center;
						center.x = bBox.x + bBox.width / 2;
						center.y = bBox.y + bBox.height / 2;		         
			    	
				    	int diff_x = center.x - noFoundStartPt.x;
				    	int diff_y = center.y - noFoundStartPt.y;
				    	int distance  = diff_x * diff_x + diff_y * diff_y;

						if( distance < closest_dist)
						{
							closest_dist = distance;
							best_ball_center = center;
							best_ball_box    = bBox;
							ballFound = true;						
						}
				    }
				    
				    if(ballFound)
				    {
	    			    //measurement.at<float>(0) = best_ball_center.x;  
						//measurement.at<float>(1) = best_ball_center.y;  
	    				//Mat estimated = KF.correct(measurement);	
						KF.statePost.at<float>(0) = best_ball_center.x;
						KF.statePost.at<float>(1) = best_ball_center.y;
						KF.statePost.at<float>(2) = 0;
						KF.statePost.at<float>(3) = 0;

						prev_box     = best_ball_box;
						prev_motion  = Point2f(0, 0);
				    	noFoundCount = 0;
				    }
				    else {
				    	cout << "UNABLE TO CORRECT..." << endl;
				    }
				}
				noFoundCount++;
				cout << "NO FOUND: " << noFoundCount << endl;
		    }
		    
		    // rendering result
			line( result, statePt, predictPt, CV_RGB(255,0,255), 2 );	
	    	circle( result, predictPt, 2, CV_RGB(255,0,255), 2 );	         
		    circle( result, best_ball_center, 2, CV_RGB(255,255,255), 2 );
		    rectangle( result, best_ball_box, CV_RGB(0,255,0), 2 );

			// Optical Flow   
            /*
            for (size_t i = 0; i < optFlow_ball_centers.size(); i++)
			{
				line( result, prev_ball_centers[i], optFlow_ball_centers[i], CV_RGB(120,70,255), 2 );
		    	circle( result, optFlow_ball_centers[i], 2, CV_RGB(120,70,255), 2 );
            }			   
			*/
			
			// Hough
            for( size_t circle_i = 0; circle_i < circles.size(); circle_i++ )
            {                
                Point center(cvRound(circles[circle_i][0]), cvRound(circles[circle_i][1]));
                int radius = cvRound(circles[circle_i][2]);
                circle( result, center, radius, Scalar(12,12,255), 2 );
            }			
			
			prev_ball_centers = cur_contour_centers;
			
		    imshow("Result Window", result);
            
            /* UPDATE FRAME */
            cur_frame.copyTo( prev_frame );
            
            /* KEY INPUTS */
            int keynum = waitKey(30) & 0xFF;
            if(keynum == 113)      // press q
            	break;
            else if(keynum == 32)  // press space
			{
				keynum = 0;
				while(keynum != 32 && keynum != 113)
					keynum = waitKey(30) & 0xFF;
				if(keynum == 113) 
					break;
			}
        }
        inputVideo.release();
        outputVideo.release();
    }
    else {
        cout << "Not enought parameters! \n";
        return -1;
    }
    
    return 0;
}
