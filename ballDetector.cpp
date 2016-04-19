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

		//namedWindow("Basketball", CV_WINDOW_AUTOSIZE);		
        //namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

		int iLowH = 0;
		int iHighH = 5;

		int iLowS = 0; 
		int iHighS = 255;

		int iLowV = 0;
		int iHighV = 255;
		/*
		//Create trackbars in "Control" window
		cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
		cvCreateTrackbar("HighH", "Control", &iHighH, 179);

		cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
		cvCreateTrackbar("HighS", "Control", &iHighS, 255);

		cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
		cvCreateTrackbar("HighV", "Control", &iHighV, 255);
		*/
	        
		for(int i=0; i < inputVideo.get(CAP_PROP_FRAME_COUNT); ++i)
		{
		    Mat frame, frameHSV;
		    inputVideo >> frame; // get a new frame from camera

			cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert the captured frame from BGR to HSV

			Mat imgThresholded;

			inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); // Threshold the image
			  
			// morphological opening (remove small objects from the foreground)
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
			dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

			// morphological closing (fill small holes in the foreground)
			dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

			imshow("Thresholded Image", imgThresholded); //show the thresholded image

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
