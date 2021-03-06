#include <iostream> // for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    argc = 3;
    if(argc >= 3)
    {
        const string inputfile      = "/Users/ikehuang/Documents/COURSE/digital_video/ball_detection/digital_video_final/video-1461463164.mp4";
        VideoCapture inputVideo(inputfile); // open the default camera
        if(!inputVideo.isOpened())  // check if we succeeded
            return -1;
        
        // Initialize
        VideoWriter outputVideo;  // Open the output
        const string source      = "/Users/ikehuang/Desktop/ball_video/output";
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
        
        //namedWindow("Basketball", CV_WINDOW_AUTOSIZE);
        //namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
        
        int iLowH = 5;
        int iHighH = 40;
        
        int iLowS = 100;
        int iHighS = 255;
        
        int iLowV = 70;
        int iHighV = 140;
        
        for(int i=0; i < inputVideo.get(CAP_PROP_FRAME_COUNT); ++i)
        {
            Mat frame, frameHSV, frameGray;
            inputVideo >> frame; // get a new frame from camera
            
            cv::GaussianBlur(frame, frame, cv::Size(5, 5), 3.0, 3.0);
            
            cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert the captured frame from BGR to HSV
            
            Mat mask1;
            
            inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mask1);   // Threshold the image
            
            // morphological opening (remove small objects from the foreground)
            erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            
            // morphological closing (fill small holes in the foreground)
            dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            
            Mat frameFiltered, frameGray2;
            frame.copyTo( frameFiltered, mask1 );
            
            /*if(i == 10)
             {
             vector<int> compression_params;
             compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
             compression_params.push_back(9);
             imwrite("test.png", frame, compression_params);
             }*/
            
            cv::cvtColor( frameFiltered, frameGray2, CV_BGR2GRAY );
            vector<cv::Vec3f> circles;
            cv::GaussianBlur(frameGray2, frameGray2, cv::Size(5, 5), 3.0, 3.0);
            
            HoughCircles( frameGray2, circles, CV_HOUGH_GRADIENT, 1, 60, 200, 20, 10, 30 );
            
            Mat result = frameFiltered;
            
            for( size_t i = 0; i < circles.size(); i++ )
            {
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                cv::circle( result, center, 3, Scalar(0,255,255), -1);
                cv::circle( result, center, radius, Scalar(0,0,255), 1 );
            }
            
            /*vector< vector<cv::Point> > contours;
             cv::findContours(mask1, contours, CV_RETR_EXTERNAL,
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
             if (ratio > 0.75 && bBox.area() >= 400)
             {
             balls.push_back(contours[i]);
             ballsBox.push_back(bBox);
             }
             }
             
             Mat result = frameFiltered;
             for (size_t i = 0; i < balls.size(); i++)
             {
             cv::drawContours(result, balls, i, CV_RGB(20,150,20), 1);
             cv::rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
             
             cv::Point center;
             center.x = ballsBox[i].x + ballsBox[i].width / 2;
             center.y = ballsBox[i].y + ballsBox[i].height / 2;
             cv::circle(result, center, 2, CV_RGB(20,150,20), -1);
             
             stringstream sstr;
             sstr << "(" << center.x << "," << center.y << ")";
             cv::putText(result, sstr.str(),
             cv::Point(center.x + 3, center.y - 3),
             cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
             }
             */
            imshow("Filtered Image", result); //show the thresholded image
            
            if(waitKey(30) >= 0) break;
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
