#include <iostream> // for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		Mat* img = (Mat*)userdata;  // 1st cast it back, then deref
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		cout << img->at<Vec3b>( Point(x, y) ) << endl;
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
        int iLowH = 180;
        int iHighH = 20;
        
        int iLowS = 50;
        int iHighS = 225;
        
        int iLowV = 50;
        int iHighV = 160;

        // Field Color
        int fieldLowH = 0;
        int fieldHighH = 20;
        
        int fieldLowS = 50;
        int fieldHighS = 150;
        
        int fieldLowV = 160;
        int fieldHighV = 255;
 
		namedWindow("Result Window", 1);
		
		// Mat declaration
		Mat prev_frame, prev_gray, cur_frame, cur_gray;
        Mat frame_blurred, frameHSV, frameGray;
        
        // take the first frame
        inputVideo >> prev_frame;
		
        // field boundary
		Point l_top( prev_frame.cols/2, prev_frame.rows );
		Point l_bot( prev_frame.cols/2, 0 );
		Point r_top( prev_frame.cols/2, prev_frame.rows );
		Point r_bot( prev_frame.cols/2, 0 );
		Point m_top_l( prev_frame.cols/2, prev_frame.rows );
		Point m_top_r( prev_frame.cols/2, prev_frame.rows );
		Point field_bound[6];
        
        for(int frame_num=1; frame_num < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_num)
        {
            inputVideo >> cur_frame; // get a new frame from camera
            
            // Blur & convert frame to HSV color space
            cv::GaussianBlur(prev_frame, frame_blurred, cv::Size(5, 5), 3.0, 3.0);
            cvtColor(frame_blurred, frameHSV, COLOR_BGR2HSV);
            
            // gray scale current frame
    		cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);            
    		cvtColor(cur_frame, cur_gray, CV_BGR2GRAY);
            
            // creating masks
            Mat mask, mask1, mask2, field_mask;
            inRange(frameHSV, Scalar(0, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);
            inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2);
            inRange(frameHSV, Scalar(fieldLowH, fieldLowS, fieldLowV), Scalar(fieldHighH, fieldHighS, fieldHighV), field_mask);
            
            mask = mask1 + mask2;
            
            // morphological opening (remove small objects from the foreground)
            erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            // morphological closing (fill small holes in the foreground)
            //dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            //erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
			
            // morphological opening (remove small objects from the foreground)
            erode(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
                         
            //Mat frameFiltered;
            //cur_frame.copyTo( frameFiltered, mask );
                        
            vector< vector<cv::Point> > contours_ball;
            vector< vector<cv::Point> > contours_field;           
            cv::findContours(mask, contours_ball, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            cv::findContours(field_mask, contours_field, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            
            Mat result = prev_frame;
			
			// update field boundary every 8 frames
			if(frame_num % 8 == 1)
			{		
				l_top = Point( prev_frame.cols/2, prev_frame.rows );
				l_bot = Point( prev_frame.cols/2, 0 );
				r_top = Point( prev_frame.cols/2, prev_frame.rows );
				r_bot = Point( prev_frame.cols/2, 0 );
				m_top_l = Point( prev_frame.cols/2, prev_frame.rows );
				m_top_r = Point( prev_frame.cols/2, prev_frame.rows );
				
		        for (size_t i = 0; i < contours_field.size(); i++)
		        {
				    Rect bBox;
				    bBox = cv::boundingRect(contours_field[i]);
	 				// don't consider field contours that are too small
	 				if( bBox.area() < 900 )
						continue;
	 		         
	 		        int l_bound = bBox.x;
	 		        int r_bound = bBox.x + bBox.width;
	 		        int u_bound = bBox.y;
	 		        int b_bound = bBox.y + bBox.height;
	 		         
	 		        // if we found a more left boundary 
	 		        if( l_bound <= l_top.x )
	 		        {
		 		    	if( l_bound == l_top.x )
	 		         	{
	 		         		if( u_bound < l_top.y )
	 		         			l_top = Point( l_bound, u_bound );
	 		         		if( b_bound > l_bot.y )
		 		         		l_bot = Point( l_bound, b_bound );
	 		         	}
	 		         	else  // overwrite ltop, lbot
	 		         	{
	 		         		l_top = Point( l_bound, u_bound );
	 		         		l_bot = Point( l_bound, b_bound );
	 		        	}
	 		        }
	 		         
	 		        // if we found a more right boundary 
	 		        if( r_bound >= r_top.x )
	 		        {
		 		    	if( r_bound == r_top.x )
	 		         	{
	 		         		if( u_bound < r_top.y )
	 		         			r_top = Point( r_bound, u_bound );
	 		         		if( b_bound > r_bot.y )
		 		         		r_bot = Point( r_bound, b_bound );
	 		         	}
	 		        	else  // overwrite rtop, rbot
	 		        	{
	 		         		r_top = Point( r_bound, u_bound );
	 		         		r_bot = Point( r_bound, b_bound );
	 		        	}
	 		        }
	
	 		        // found a more upper boundary
	 		        if( u_bound < m_top_l.y )
	 		        {
	 		        	m_top_l = Point( l_bound, u_bound );
	 		        	m_top_r = Point( r_bound, u_bound ); 		        	
	 		        }	
				} 		         
            }    
			
			field_bound[0] = l_top;
			field_bound[1] = l_bot;
			field_bound[2] = r_bot;
			field_bound[3] = r_top;
			field_bound[4] = m_top_r;
			field_bound[5] = m_top_l;
			
			Mat field( mask.rows, mask.cols, CV_8UC1, Scalar(0) );
			fillConvexPoly( field, field_bound, 6, 255);
			
			cv::line( result, l_top,   l_bot,   CV_RGB(0,255,255), 2);	
			cv::line( result, l_bot,   r_bot,   CV_RGB(0,255,255), 2);	
			cv::line( result, r_bot,   r_top,   CV_RGB(0,255,255), 2);	
			cv::line( result, r_top,   m_top_r, CV_RGB(0,255,255), 2);	
			cv::line( result, m_top_r, m_top_l, CV_RGB(0,255,255), 2);
			cv::line( result, m_top_l, l_top,   CV_RGB(0,255,255), 2);

            // sieves
            vector< vector<cv::Point> > balls;
            vector<cv::Point2f> prev_ball_centers;
            vector<cv::Rect> ballsBox;
            for (size_t i = 0; i < contours_ball.size(); i++)
			{
			    drawContours(result, contours_ball, i, CV_RGB(255,0,0), 1);  // fill the area
			     
		        cv::Rect bBox;
		        bBox = cv::boundingRect(contours_ball[i]);

		        // ball size sieve
		        if( bBox.area() < 200 || bBox.area() > 1000 ) 
		        	continue;
		         	
		        // ratio sieve
		        float ratio = (float) bBox.width / (float) bBox.height;
				if( ratio < 1.0/3.0 || ratio > 3.0 )
				 	continue;
		         
                // ball center sieve: since we've done dilate and erode, not necessary to do.
		        cv::Point center;
		        center.x = bBox.x + bBox.width / 2;
		        center.y = bBox.y + bBox.height / 2;
		        		        					
				/*
				uchar center_v = mask.at<uchar>( center );*
				if(center_v != 1)
			    	continue;
			  	*/ 
			  	 
			  	// ball-on-court assumption 
			  	//if(field.at<uchar>(center) != 255)
				//	continue;
									 
			    balls.push_back(contours_ball[i]);
				prev_ball_centers.push_back(center);
				ballsBox.push_back(bBox);
            }

			vector<cv::Point2f> cur_ball_centers;
            vector<uchar> featuresFound;
            Mat err;
            TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
			Size winSize(31, 31);
			if( prev_ball_centers.size() > 0)
				calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_ball_centers, cur_ball_centers, featuresFound, err, winSize, 3, termcrit, 0, 0.001);
             
            // draw candidates
            for (size_t i = 0; i < balls.size(); i++)
            {
		        //cv::drawContours(result, balls, i, CV_RGB(20,150,20), 2);
		        //cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
		         
		        cv::Point2f prev_center = prev_ball_centers[i];
		        cv::Point2f curr_center = cur_ball_centers[i];
				
				if(!featuresFound[i])
		        	continue;
		        		        	
		        //cv::circle(result, cvPoint(prev_center.x, prev_center.y), 1, CV_RGB(20,150,20), -1, 8, 0);
		        //cv::circle(result, cvPoint(curr_center.x, curr_center.y), 1, CV_RGB(20,150,20), -1, 8, 0);
				cv::line( result, prev_center, curr_center, CV_RGB(255,0,0), 3);	
		         
		        stringstream sstr;
		        sstr << "(" << prev_center.x << "," << prev_center.y << ")";
		        cv::putText(result, sstr.str(),
		        cv::Point(prev_center.x + 3, prev_center.y - 3),
		        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
            }
             
 
			setMouseCallback("Result Window", CallBackFunc, &frameHSV);	
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
