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
        
        // Basketball Color 
        int iLowH = 170;
        int iHighH = 20;
        
        int iLowS = 50;
        int iHighS = 255;
        
        int iLowV = 50;
        int iHighV = 160;

		namedWindow("Result Window", 1);
		
		// Mat declaration
		Mat prev_frame, prev_gray, cur_frame, cur_gray;
        Mat frame_blurred, frameHSV, frameGray;
        
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
		float transMatrixData[16] = {1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1};
		KF.transitionMatrix = Mat(4, 4, CV_32F, transMatrixData);
		Mat_<float> measurement(2,1);
		measurement.setTo(Scalar(0));
		
		/* We put the first point in predicted state */
		KF.statePre.at<float>(0) = mp.pt.x;
		KF.statePre.at<float>(1) = mp.pt.y;
		KF.statePre.at<float>(2) = 0;
		KF.statePre.at<float>(3) = 0;
		setIdentity(KF.measurementMatrix);                        // measurement matrix H
		setIdentity(KF.processNoiseCov, Scalar::all(1e-4));       // process noise covariance matrix Q
		setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));   // measurement noise covariance matrix R
		setIdentity(KF.errorCovPost, Scalar::all(.1));            // posteriori error estimate cov matrix P(t)
	
		/* Some extra params */
        Rect  prev_BallBox;
		Point prev_BallCenter;
		Point prev_Motion;

        /* start tracking */		
		setMouseCallback("Result Window", CallBackFunc, &frameHSV);	
		
        for(int frame_num=1; frame_num < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_num)
        {
        	/* get current frame */
            inputVideo >> cur_frame;
            
            /* Kalman Filter Prediction */
			Mat prediction = KF.predict();
			//Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
			// Kalman Filter Update
			//Mat estimated = KF.correct( best_candidate );

            // Blur & convert frame to HSV color space
            cv::GaussianBlur(prev_frame, frame_blurred, cv::Size(5, 5), 3.0, 3.0);
            cvtColor(frame_blurred, frameHSV, COLOR_BGR2HSV);
            
            // gray scale current frame
    		cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);            
    		cvtColor(cur_frame, cur_gray, CV_BGR2GRAY);
            
            /* 
             * STAGE 1: mask generation
             * creating masks for balls and courts.
             */   
            Mat mask, mask1, mask2, court_mask;
            inRange(frameHSV, Scalar(0, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);
            inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2);
            inRange(frameHSV, Scalar(courtLowH, courtLowS, courtLowV), Scalar(courtHighH, courtHighS, courtHighV), court_mask);
            
            mask = mask1 + mask2;
            
            // morphological opening (remove small objects from the foreground)
            erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
			

            /* 
             * STAGE 2: contour generation
             * creating contours with masks.
             */   			
            vector< vector<cv::Point> > contours_ball;
            vector< vector<cv::Point> > contours_court;
            cv::findContours(mask, contours_ball, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            
            Mat result;
            prev_frame.copyTo( result );
			
            // sieves
            vector< vector<cv::Point> > balls;
            vector<cv::Point2f> prev_ball_centers;
            vector<cv::Rect> ballsBox;
            Point best_candidate;
            for (size_t i = 0; i < contours_ball.size(); i++)
			{
			    drawContours(result, contours_ball, i, CV_RGB(255,0,0), 1);  // fill the area
			     
		        cv::Rect bBox;
		        bBox = cv::boundingRect(contours_ball[i]);
			    Point center;
			    center.x = bBox.x + bBox.width / 2;
			    center.y = bBox.y + bBox.height / 2;		         

		        // meet prediction!
				if( mp.pt.x > bBox.x && mp.pt.x < bBox.x + bBox.width &&
				    mp.pt.y > bBox.y && mp.pt.y < bBox.y + bBox.height)
				{
					// initialization of ball position at first frame
					if( frame_num == 1 || ( bBox.area() <= lastBallBox.area() * 1.5 && bBox.area() >= lastBallBox.area() * 0.5) )
					{
						lastBallBox = bBox;
						lastBallCenter = center;

						balls.push_back(contours_ball[i]);
						prev_ball_centers.push_back(center);
						ballsBox.push_back(bBox);
						best_candidate = center;
					}
					else
					{
						cout << "area changed!" << endl;
						// if the block containing ball becomes too large,
						// we use last center + motion as predicted center
						balls.push_back(contours_ball[i]);
						prev_ball_centers.push_back( lastBallCenter+lastMotion );
						ballsBox.push_back(bBox);
						best_candidate = lastBallCenter + lastMotion; 
					}
				}
				else
				{
				    // ball size sieve
				    if( bBox.area() < 100 || bBox.area() > 1600 ) 
				    	continue;
				     	
				    // ratio sieve
				    float ratio = (float) bBox.width / (float) bBox.height;
					if( ratio < 1.0/3.0 || ratio > 3.0 )
					 	continue;
					
		            // ball center sieve: since we've done dilate and erode, not necessary to do.
					
					uchar center_v = mask.at<uchar>( center );*
					if(center_v != 1)
						continue;
				  	 
				  	// ball-on-court sieve: not useful in basketball =( 
				  	//if(court_mask.at<uchar>(center) != 255)
					//	continue;
					
					balls.push_back(contours_ball[i]);
					prev_ball_centers.push_back(center);
					ballsBox.push_back(bBox);					
				}
            }

			vector<Point2f> cur_ball_centers;
            vector<uchar> featuresFound;
            Mat err;
            TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
			Size winSize(31, 31);
			if( prev_ball_centers.size() > 0 )
				calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_ball_centers, cur_ball_centers, featuresFound, err, winSize, 3, termcrit, 0, 0.001);
			
		    circle(result, mp.pt, 2, CV_RGB(255,255,255), 2);
			
            bool ball_found = false;
            for (size_t i = 0; i < balls.size(); i++)
            {
            	// see if any candidates contains out ball
				if( mp.pt.x > ballsBox[i].x && mp.pt.x < ballsBox[i].x + ballsBox[i].width && 
				    mp.pt.y > ballsBox[i].y && mp.pt.y < ballsBox[i].y + ballsBox[i].height)
				{
			        cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
			        Point motion = cur_ball_centers[i] - prev_ball_centers[i];
			        // update points and lastMotion
			        mp.pt = Point2f(mp.pt.x+motion.x, mp.pt.y+motion.y);  // TODO replace with predicted points of kalman filter here.
			        lastMotion = motion;
			        ball_found = true;
				}
				
				// draw optical flow
				if(!featuresFound[i])
	        		continue;
		         
		        cv::Point2f prev_center = prev_ball_centers[i];
		        cv::Point2f curr_center = cur_ball_centers[i];		        		        	
				cv::line( result, prev_center, curr_center, CV_RGB(255,255,0), 2);			         
            }
            
            // if ball is not found, search for the closest ball candidate within a distance.
			if(!ball_found)
			{
				int search_distance_threshold = 60*60;
				int closest_dist      = 10000;
				int closest_area_diff = 10000;
				int best_i = 0;
				
		        for (size_t i = 0; i < balls.size(); i++)
		        {
		        	int diff_x = prev_ball_centers[i].x - mp.pt.x;
		        	int diff_y = prev_ball_centers[i].y - mp.pt.y;
		        	int distance  = diff_x * diff_x + diff_y * diff_y;
					int area_diff = abs(ballsBox[i].area()-lastBallBox.area());
					// if distance is small
					if( distance < search_distance_threshold &&
					    distance < closest_dist && 
					    area_diff < closest_area_diff )
					{
						closest_dist      = distance;
						closest_area_diff =  area_diff;
						best_i = i;
						ball_found = true;
					}				
		        }

		        if(ball_found)
				{
					// reset mp.pt
				    cv::rectangle(result, ballsBox[best_i], CV_RGB(255,255,0), 2);
			        mp.pt = cur_ball_centers[best_i];
			    }
			    else
				{
					// if ball still not found... stay at the same direction
				    circle(result, mp.pt, 5, CV_RGB(255,255,255), 2);
				    mp.pt = lastBallCenter + lastMotion;
				}
			}
			   
			   
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
