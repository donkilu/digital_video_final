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
        
        // BASKETBALL
        int iLowH = 180;
        int iHighH = 16;
        
        int iLowS = 100;
        int iHighS = 255;
        
        int iLowV = 70;
        int iHighV = 160;
        
        // Field Color
        int fieldLowH = 0;
        int fieldHighH = 20;
        
        int fieldLowS = 50;
        int fieldHighS = 150;
        
        int fieldLowV = 160;
        int fieldHighV = 255;
        
        namedWindow("My Window", 1);
        vector<cv::Point> point_history_tmp[40];
        
        for(int frame_i=0; frame_i < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_i)
        {
            Mat frame, frameHSV, frameGray;
            inputVideo >> frame; // get a new frame from camera
            
            cv::GaussianBlur(frame, frame, cv::Size(5, 5), 3.0, 3.0);
            
            cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert the captured frame from BGR to HSV
            
            Mat mask, mask1, mask2, field_mask,frameFiltered, frameGray2;
            
            // creating masks
            inRange(frameHSV, Scalar(1, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);
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
            /*
             // morphological closing (fill small holes in the foreground)
             dilate(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
             erode(field_mask, field_mask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
             */

            frame.copyTo( frameFiltered, mask );
            
            vector< vector<cv::Point> > contours_ball;
            vector< vector<cv::Point> > contours_field;
            cv::findContours(mask, contours_ball, CV_RETR_EXTERNAL,
                             CV_CHAIN_APPROX_NONE);
            cv::findContours(field_mask, contours_field, CV_RETR_EXTERNAL,
                             CV_CHAIN_APPROX_NONE);
            
            /* draw result */
            frame.copyTo( frameFiltered, mask );
            
            cv::cvtColor( frameFiltered, frameGray2, CV_BGR2GRAY );
            vector<cv::Vec3f> circles;
            cv::GaussianBlur(frameGray2, frameGray2, cv::Size(5, 5), 2.0, 2.0);
            
            HoughCircles( frameGray2, circles, CV_HOUGH_GRADIENT, 1, frameGray2.rows/8, 90, 8, 0,  50 );
            
            Mat result = frame;
            
            // pick field boundary
            Point l_top( mask.cols/2, mask.rows );
            Point l_bot( mask.cols/2, 0 );
            Point r_top( mask.cols/2, mask.rows );
            Point r_bot( mask.cols/2, 0 );
            Point m_top_l( mask.cols/2, mask.rows );
            Point m_top_r( mask.cols/2, mask.rows );
            
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
            
            Point field_bound[6];
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
            vector<vector<cv::Point> > balls;
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
                 uchar center_v = mask.at<uchar>( center );
                 if(center_v != 1)
                 continue;
                 */
                
                // ball-on-court assumption
                if(field.at<uchar>(center) != 255)
                    continue;
                
                balls.push_back(contours_ball[i]);
                ballsBox.push_back(bBox);
            }
            
            
            // draw candidates
            vector<cv::Point> history_total;
            if(frame_i>39){
                for (int j = 39; j > 0; j--)
                {
                    point_history_tmp[j] = point_history_tmp[j - 1];
                }
                for (int k=1; k<40;k++){
                    history_total.insert(history_total.end(), point_history_tmp[k].begin(), point_history_tmp[k].end());
                }
            }
            
            // update the index 0, calculate the average center from previous frames
            point_history_tmp[0].clear();
            double avg_x=0, avg_y=0;
            for (size_t i = 0; i < history_total.size(); i++){
                avg_x += history_total[i].x;
                avg_y += history_total[i].y;
            }
            avg_x = avg_x/history_total.size();
            avg_y = avg_y/history_total.size();
            
            cv::Point history_center;
            history_center.x =avg_x;
            history_center.y =avg_y;
            
            // find the min distance from history center
            double dis_history_min = 1000;
            double dis_history;
            // initailize the final decision and radius
            cv::Point final_center;
            final_center.x=0; final_center.y=0;
            int final_radius=0;
            vector<double> distance_with_history;
            
            for (size_t i = 0; i < balls.size(); i++)
            {
                //                cv::drawContours(result, balls, i, CV_RGB(20,150,20), 2);
                //                cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
                //
                cv::Point center;
                center.x = ballsBox[i].x + ballsBox[i].width / 2;
                center.y = ballsBox[i].y + ballsBox[i].height / 2;
                cv::circle(result, center, 2, CV_RGB(20,150,20), -1);
                
                for( size_t i = 0; i < circles.size(); i++ )
                {
                    Point center2(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    double dis_center = cv::norm(center2-center);
                    
                    if( dis_center<radius){
                        //                            cv::drawContours(result, balls, i, CV_RGB(20,150,2), 2);
                        //                            cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
                        
                        if(frame_i<40){
                            point_history_tmp[frame_i].push_back(center);
                        }else{
                            point_history_tmp[0].push_back(center);
                        }
                        dis_history = cv::norm(history_center-center);
                        if(dis_history<dis_history_min){
                            dis_history_min=dis_history;
                            final_center =center;
                            final_radius =radius;
                        }
                        //                            stringstream sstr;
                        //                            sstr << "(" << center.x << "," << center.y << ")";
                        //                            cv::putText(result, sstr.str(),
                        //                                    cv::Point(center.x + 3, center.y - 3),
                        //                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
                        //                            cv::circle( result, center, 3, Scalar(128,128,128), 5);
                        //                            cv::circle( result, center, radius, Scalar(128,128,255), 5 );
                    }
                    
                }
                
            }
            stringstream sstr;
            sstr << "(" << final_center.x << "," << final_center.y << ")";
            cv::putText(result, sstr.str(),
                        cv::Point(final_center.x + 3, final_center.y - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
            cv::circle( result, final_center, 3, Scalar(128,128,128), 5);
            cv::circle( result, final_center, final_radius, Scalar(128,128,255), 5 );
            
            history_total.clear();
            
            
            
            setMouseCallback("My Window", CallBackFunc, &frameHSV);
            imshow("My Window", result);
            
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
    
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
