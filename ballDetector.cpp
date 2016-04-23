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
		int iHighH = 10;

		int iLowS = 0; 
		int iHighS = 255;

		int iLowV = 0;
		int iHighV = 128;
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
		    Mat frame, frameHSV, frameGray;
		    inputVideo >> frame; // get a new frame from camera

			cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert the captured frame from BGR to HSV

			Mat mask, mask1, mask2;

			inRange(frameHSV, Scalar(0, iLowS, iLowV), Scalar(25, iHighS, iHighV), mask1);      // Threshold the image
			//inRange(frameHSV, Scalar(170, iLowS, iLowV), Scalar(180, iHighS, iHighV), mask2); // Threshold the image
			  
			// morphological opening (remove small objects from the foreground)
			erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
			dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		
			//erode(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
			//dilate(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

			// morphological closing (fill small holes in the foreground)
			dilate(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
			erode(mask1, mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

			//dilate(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
			//erode(mask2, mask2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

			mask = mask1 + mask2;

			Mat frameFiltered;
			frame.copyTo( frameFiltered, mask );
			
			if(i == 10) 
			{			
				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);
				imwrite("test.png", frame, compression_params);
			}
			
			/* TODO Other Transform */
			/*Mat detectedEdge;
			/// Convert the image to grayscale
			cvtColor( frame, frameGray, CV_BGR2GRAY );
			/// Reduce noise with a kernel 3x3
			blur( frameGray, detectedEdge, Size(3,3) );

			/// Canny detector
			int minTh = 50;
			int kernel_size = 3;
			Canny( detectedEdge, detectedEdge, minTh, minTh*2, kernel_size );
			*/
			/// Using Canny's output as a mask, we display our result
			//dst = Scalar::all(0);

			//src.copyTo( dst, detected_edges);
			//imshow( window_name, dst );
			
			imshow("Filtered Image", frameFiltered); //show the thresholded image

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
