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
        
        namedWindow("My Window", 1);
        vector<cv::Point> point_history_tmp[40];  // previos frames's vector, containing "center"
        
        
        for(int frame_i=0; frame_i < inputVideo.get(CAP_PROP_FRAME_COUNT); ++frame_i)
        {
            Mat frame, frameHSV, frameGray;
            inputVideo >> frame; // get a new frame from camera
            
            cv::GaussianBlur(frame, frame, cv::Size(5, 5), 3.0, 3.0);
            
            cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert the captured frame from BGR to HSV
            
            Mat mask1, mask2;
            
            inRange(frameHSV, Scalar(2, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);   // Threshold the image
            inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2);  // Threshold the image
            
            // morphological opening (remove small objects from the foreground)
            erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            erode(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            
            Mat mask, frameFiltered, frameGray2;
            mask = mask1 + mask2;
            frame.copyTo( frameFiltered, mask );
            
            cv::cvtColor( frameFiltered, frameGray2, CV_BGR2GRAY );
            vector<cv::Vec3f> circles;
            cv::GaussianBlur(frameGray2, frameGray2, cv::Size(5, 5), 2.0, 2.0);
            
            HoughCircles( frameGray2, circles, CV_HOUGH_GRADIENT, 1, frameGray2.rows/8, 150, 18, 5,  150 );
            
            Mat result = frame;
            
//            for( size_t i = 0; i < circles.size(); i++ )
//            {
//                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//                int radius = cvRound(circles[i][2]);
//                cv::circle( result, center, 3, Scalar(128,128,128), 5);
//                cv::circle( result, center, radius, Scalar(128,128,255), 5 );
//            }
            
            
            vector< vector<cv::Point> > contours;
            cv::findContours(mask, contours, CV_RETR_EXTERNAL,
                             CV_CHAIN_APPROX_NONE);
            
            // >>>>> Filtering

            vector<vector<cv::Point> > balls;
            vector<cv::Rect> ballsBox;
            for (size_t i = 0; i < contours.size(); i++)
            {
                cv::Rect bBox;
                bBox = cv::boundingRect(contours[i]);
                
                float ratio = (float) bBox.width / (float) bBox.height;
                if (ratio > 1.0f)
                    ratio = 1.0f / ratio;
                
                // Searching for a bBox almost square
                if (ratio > 0.7 && bBox.area() > 200)
                {
                    balls.push_back(contours[i]);
                    ballsBox.push_back(bBox);
                }
//                drawContours(result, contours, i, CV_RGB(255,0,0), 1);
            }
            
            
            // combining point_history_tmp, from index 1 to end
            // index 0 does not count in the history_total
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
            cout<<"size after:"<<point_history_tmp[0].size()<<endl;
            
            
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
