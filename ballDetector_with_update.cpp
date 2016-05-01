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
        int iLowH = 150;
        int iHighH = 16;
        
        int iLowS = 70;
        int iHighS = 170;
        
        int iLowV = 70;
        int iHighV = 150;
        
        // court Color
        int courtLowH = 0;
        int courtHighH = 20;
        
        int courtLowS = 50;
        int courtHighS = 150;
        
        int courtLowV = 160;
        int courtHighV = 255;
        
        namedWindow("Result Window", 1);
        //namedWindow("Court Window", 1);
        
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
        Rect  lastBallBox;
        Point lastBallCenter;
        Point lastMotion;
        
        /* Kalman Filter Initialization */
        KalmanFilter KF(4, 2, 0);
        float transMatrixData[16] = {1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1};
        KF.transitionMatrix = Mat(4, 4, CV_32F, transMatrixData);
        Mat_<float> measurement(2,1);
        measurement.setTo(Scalar(0));
        
        KF.statePre.at<float>(0) = mp.pt.x;
        KF.statePre.at<float>(1) = mp.pt.y;
        KF.statePre.at<float>(2) = 0;
        KF.statePre.at<float>(3) = 0;
        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(.1));
        
        /* start tracking */
        setMouseCallback("Result Window", CallBackFunc, &frameHSV);
        
        for(int frame_num=1; frame_num < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_num)
        {
            inputVideo >> cur_frame; // get a new frame
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
            inRange(frameHSV, Scalar(1, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);
            inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2);
            inRange(frameHSV, Scalar(courtLowH, courtLowS, courtLowV), Scalar(courtHighH, courtHighS, courtHighV), court_mask);
            
            mask = mask1 + mask2;
            
            // morphological opening (remove small objects from the foreground)
            erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            
            /*
             * Method:  HoughCircles
             * creating circles and radius.
             */
            // Basketball Color for Hough circle
            
            int iLowH = 150;
            int iHighH = 16;
            
            int iLowS = 70;
            int iHighS = 165;
            
            int iLowV = 70;
            int iHighV = 150;
            
            Mat mask1_circle, mask2_circle, mask_circle, frameHSV_circle, frameFiltered,frameGray2;
            cvtColor(frame_blurred, frameHSV_circle, COLOR_BGR2HSV);
            inRange(frameHSV_circle, Scalar(2, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1_circle);
            inRange(frameHSV_circle, Scalar(iLowH, iLowS, iLowV),Scalar(180, iHighS, iHighV), mask2_circle);
            mask_circle = mask1_circle + mask2_circle;
            erode(mask_circle, mask_circle, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            dilate(mask_circle, mask_circle, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
            
            prev_frame.copyTo( frameFiltered, mask_circle );
            cv::cvtColor( frameFiltered, frameGray2, CV_BGR2GRAY );
            vector<cv::Vec3f> circles;
            cv::GaussianBlur(frameGray2, frameGray2, cv::Size(5, 5), 3.0, 3.0);
            HoughCircles( frameGray2, circles, CV_HOUGH_GRADIENT, 1, frameGray2.rows/8, 80, 16, 5,400);
            
            /*
             * STAGE 2: contour generation
             * creating contours with masks.
             */
            vector< vector<cv::Point> > contours_ball;
            vector< vector<cv::Point> > contours_court;
            cv::findContours(mask, contours_ball, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            
            Mat result;

            prev_frame.copyTo( result );
        
            /*
             // court mask refinement: eliminate small blocks
             Mat buffer;
             court_mask.copyTo( buffer );
             cv::findContours(buffer, contours_court, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
             
             for (size_t i = 0; i < contours_court.size(); i++)
             {
	            double tmp_area = contourArea( contours_court[i] );
	            if(tmp_area < 900.0)
             drawContours(court_mask, contours_court, i, 0, CV_FILLED);
             }
             bitwise_not(court_mask, court_mask);
             court_mask.copyTo( buffer );
             cv::findContours(buffer, contours_court, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
             for (size_t i = 0; i < contours_court.size(); i++)
             {
	            double tmp_area = contourArea( contours_court[i] );
	            if(tmp_area < 900.0)
             drawContours(court_mask, contours_court, i, 0, CV_FILLED);
             }
             bitwise_not(court_mask, court_mask);
             
             Mat canny_mask;
             Canny(court_mask, canny_mask, 50, 150, 3);
             vector<Vec4i> lines;
             HoughLinesP(canny_mask, lines, 1, CV_PI/180, 80, 30, 10);
             
             Point l_top( mask.cols/2, mask.rows );
             Point l_bot( mask.cols/2, mask.rows );
             
             for( size_t i = 0; i < lines.size(); i++ )
             {
             Point p1 = Point(lines[i][0], lines[i][1]);
             Point p2 = Point(lines[i][2], lines[i][3]);
             
             if(p1.y < l_top.y)
             {
             l_top = p1;
             l_bot = p2;
             }
             if(p2.y < l_top.y)
             {
             l_top = p2;
             l_bot = p1;
             }
             }
             // stretch the line
             Point v_diff = l_top - l_bot;
             Point p_left, p_right;
             
             
             int left_t  = l_top.x / v_diff.x;
             int right_t = (mask.cols - l_top.x) / v_diff.x;
             
             p_left = l_top - v_diff * left_t;
             p_right = l_top + v_diff * right_t;
             
             line( court_mask, p_left, p_right, Scalar(128), 2, 8 );
             imshow("Court Window", court_mask);
             */
            
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
                  
                     if(  bBox.area() > 1600 )
                     continue;
                     
                     // ratio sieve
//                     float ratio = (float) bBox.width / (float) bBox.height;
//                     if( ratio < 1.0/2.0 || ratio > 2.0 )
//                     continue;
                    
                     // ball center sieve: since we've done dilate and erode, not necessary to do.
                     /*
                     uchar center_v = mask.at<uchar>( center );*
                     if(center_v != 1)
                     continue;
                     */
                    
                    // ball-on-court sieve: not useful in basketball =(
                    //if(court_mask.at<uchar>(center) != 255)
                    //	continue;
                    
                    balls.push_back(contours_ball[i]);
                    prev_ball_centers.push_back(center);
                    ballsBox.push_back(bBox);
                }
            }
            
            
            // store the center of the hough circle
            vector<cv::Point2f> prev_ball_centers_circle;
            for( size_t circle_i = 0; circle_i < circles.size(); circle_i++ )
            {
                Point center_circle(cvRound(circles[circle_i][0]), cvRound(circles[circle_i][1]));
                int radius_circle = cvRound(circles[circle_i][2]);
                prev_ball_centers_circle.push_back(center_circle);
            }
            // Kalman Filter Prediction
            //Mat prediction = KF.predict();
            //Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
            // Kalman Filter Update
            //Mat estimated = KF.correct( best_candidate );
            
            //OpticalFlow for HSV
            vector<Point2f> cur_ball_centers;
            vector<uchar> featuresFound;
            Mat err;
            TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
            Size winSize(31, 31);
            if( prev_ball_centers.size() > 0 )
                calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_ball_centers, cur_ball_centers, featuresFound, err, winSize, 3, termcrit, 0, 0.001);
            
            //OpticalFlow for circle
            vector<Point2f> cur_ball_centers_circle;
            vector<uchar> featuresFound_circle;
            Mat err2;
            if( prev_ball_centers_circle.size() > 0 )
                calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_ball_centers_circle, cur_ball_centers_circle, featuresFound_circle, err2, winSize, 3, termcrit, 0, 0.001);
            
            //plot MP
            circle(result, mp.pt, 2, CV_RGB(255,255,255), 5);
            cout<<"frame_num :"<<frame_num<<endl;
            bool ball_found = false;
            for (size_t i = 0; i < balls.size(); i++)
            {
                cv::Point center;
                center.x = ballsBox[i].x + (ballsBox[i].width / 2);
                center.y = ballsBox[i].y + (ballsBox[i].height/2);
                // consider hough circle
                int circle_in_HSV=0;
                for( size_t circle_i = 0; circle_i < circles.size(); circle_i++ )
                {

                    Point center2(cvRound(circles[circle_i][0]), cvRound(circles[circle_i][1]));
                    int radius = cvRound(circles[circle_i][2]);
                    double dis_center =  sqrt(pow(center2.x-center.x,2)+pow(center2.y-center.y,2));

                    if( frame_num >2 && radius<40 && dis_center<radius+3 && mp.pt.x > ballsBox[i].x && mp.pt.x < ballsBox[i].x + ballsBox[i].width && mp.pt.y > ballsBox[i].y && mp.pt.y < ballsBox[i].y + ballsBox[i].height){
                        circle_in_HSV=1;
                        Point motion = cur_ball_centers_circle[circle_i] - prev_ball_centers_circle[circle_i];
                        mp.pt = Point2f(cur_ball_centers_circle[circle_i].x, cur_ball_centers_circle[circle_i].y);
                        cout<<mp.pt<<endl;
                        cout<<"status 1"<<endl;
                        ball_found = true;
                    }
                    if(radius<40){
                    stringstream sstr;
                    sstr << "(" << center2.x << "," << center2.y << ")";
                    cv::putText(result, sstr.str(),
                    cv::Point(center2.x + 3, center2.y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
                        cv::circle( result, center2, radius, Scalar(12,12,255), 2 );}
                }
                
                // see if any candidates contains out ball
                if( circle_in_HSV==0 && mp.pt.x > ballsBox[i].x && mp.pt.x < ballsBox[i].x + ballsBox[i].width && mp.pt.y > ballsBox[i].y && mp.pt.y < ballsBox[i].y + ballsBox[i].height && ballsBox[i].area() < 1000 && ballsBox[i].area() > 200)
                {
                    cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
                    Point motion = cur_ball_centers[i] - prev_ball_centers[i];
                    // update points and lastMotion
                    
                    float ratio = (float) ballsBox[i].width / (float) ballsBox[i].height;
                    if( ballsBox[i].area() < 1000 && ratio>0.7 && ratio<1.35 && ballsBox[i].area() > 200){
                        mp.pt = Point2f(center.x, center.y);
                        cout<<"status 2"<<endl;
                        cout<<"AREA:"<<ballsBox[i].area()<<endl;
                    }else{
                        mp.pt = Point2f(mp.pt.x+motion.x, mp.pt.y+motion.y);
                        cout<<"status 3"<<endl;
                    }
                    // TODO replace with predicted points of kalman filter here.
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
                int search_distance_threshold = 35*35;
                int closest_dist      = 1500;
//                int closest_dist2      = 2000;
                int closest_area_diff = 10000;
                int best_i = 0;
                
                for (size_t i = 0; i < balls.size(); i++)
                {
                    int diff_x = prev_ball_centers[i].x - mp.pt.x;
                    int diff_y = prev_ball_centers[i].y - mp.pt.y;
                    int area_threshold_high = 100*100;
                    int area_threshold_low = 15*15;
                    int distance  = diff_x * diff_x + diff_y * diff_y;
                    int area_diff = abs(ballsBox[i].area()-lastBallBox.area());
                    float ratio = (float) ballsBox[i].width / (float) ballsBox[i].height;
//                    if(distance<closest_dist2){
//                        closest_dist2=distance;
//                        best_i = i;}
                    // if distance is small
                    if( distance < search_distance_threshold &&
                       distance < closest_dist && ratio>0.5 && ratio<2 && ballsBox[i].area()<area_threshold_high && ballsBox[i].area()>area_threshold_low)
                    {
                        closest_dist      = distance;
                        closest_area_diff =  area_diff;
                        best_i = i;
                        ball_found = true;
                    }				
                }
//                cout<<"ballsBox[i].area()"<<ballsBox[best_i].area()<<endl;
//                cout<<"Ratio"<<(float) ballsBox[best_i].width / (float) ballsBox[best_i].height<<endl;
                
                if(ball_found)
                {
                    // reset mp.pt
                    cout<<"here! yello"<<endl;
                    
                    int search_distance_threshold = 80*80;
                    int closest_dist = 1500;
                    int best_circle_i = 0;
                    bool circle_found = false;
                    for( size_t circle_i = 0; circle_i < circles.size(); circle_i++ )
                    {
                        int diff_x = prev_ball_centers_circle[circle_i].x - mp.pt.x;
                        int diff_y = prev_ball_centers_circle[circle_i].y - mp.pt.y;
                        int distance  = diff_x * diff_x + diff_y * diff_y;
                        if( distance < search_distance_threshold)
                        {
                            closest_dist      = distance;
                            best_circle_i = circle_i;
                            circle_found = true;
                        }
                    }
                    if(circle_found){
                        cv::circle( result, cur_ball_centers_circle[best_circle_i], 3, Scalar(255,255,0), 2 );
                        mp.pt = Point2f(cur_ball_centers_circle[best_circle_i].x, cur_ball_centers_circle[best_circle_i].y);
                        cout<<"status 4"<<endl;
                    } else{
                        cv::rectangle(result, ballsBox[best_i], CV_RGB(255,255,0), 2);
                        Point motion = cur_ball_centers[best_i] - prev_ball_centers[best_i];
                        mp.pt = Point2f(cur_ball_centers[best_i].x, cur_ball_centers[best_i].y);
                        cout<<"status 5"<<endl;
                    }

                }
                else
                {
                    // if ball still not found... stay at the same direction
                    circle(result, mp.pt, 5, CV_RGB(255,255,255), 2);
                    
                    int search_distance_threshold = 80*80;
                    int closest_dist      = 1500;
                    int best_i = 0;
                    bool ball_found = false;
                    for( size_t circle_i = 0; circle_i < circles.size(); circle_i++ )
                    {
                        int radius = cvRound(circles[circle_i][2]);
                        int diff_x = prev_ball_centers_circle[circle_i].x - mp.pt.x;
                        int diff_y = prev_ball_centers_circle[circle_i].y - mp.pt.y;
                        int distance  = diff_x * diff_x + diff_y * diff_y;
                        if( distance < search_distance_threshold && radius>10 && radius<35)
                        {
                            closest_dist      = distance;
                            best_i = circle_i;
                            ball_found = true;
                            cout<<"radius"<<radius<<endl;
                        }
                    }
                    if(ball_found){
                        cv::rectangle(result, ballsBox[best_i], CV_RGB(255,255,0), 2);
                        Point motion = cur_ball_centers_circle[best_i] - prev_ball_centers_circle[best_i];
                        mp.pt = Point2f(cur_ball_centers_circle[best_i].x, cur_ball_centers_circle[best_i].y);
                        cout<<mp.pt<<endl;
                        cout<<"status 6"<<endl;
                    }else{
//                        mp.pt = lastBallCenter + lastMotion;
                        cout<<"status 7"<<endl;
                        cout<<"lastBallCenter"<<lastBallCenter<<endl;
                        cout<<"lastMotion"<<lastMotion<<endl;}
                    
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
