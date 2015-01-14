#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define CAP_WIDTH 320
#define CAP_HEIGHT 240

namespace {
    int process(VideoCapture& cap) {
        char key;
        Mat img;
        int count = 0;
        char filename[256];
        string window_name = "M = Save | Q/Esc = Exit";

        namedWindow(window_name, 1);

        for (;;) {
            // Get a frame and display
            cap >> img;

            // Leave when no more images
            if (img.empty())
                break;

            imshow(window_name, img);

            key = (char) waitKey(30);

            switch (key) {
            // Q or Esc to leave
            case 'q':
            case 'Q':
            case 27: // Esc
                return 0;
            // M to save image
            case 'm':
            case 'M':
                sprintf(filename, "IMG_SAVED%.3d.bmp", count++);
                imwrite(filename, img);
                printf("Saved %s\n", filename);
                break;
            default:
                break;
            };

        }

        return 0;
    }
}

int main(int argc, char const *argv[])
{
    string filefmt = "";
    VideoCapture cap;

    if (filefmt.empty()) {
        // Open the default camera
        cap.open(0);
        if (
            !cap.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH) || 
            !cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        ) {
            fprintf(
                stderr, "Camera does not support w=%d:h=%d\n", 
                CAP_WIDTH, CAP_HEIGHT);
        }
    }
    else {
        // Load from files
        cap.open(filefmt);
    }

    // Check if we succeeded
    if(!cap.isOpened()) {
        fprintf(stderr, "%s\n", "");
        cerr << "No camera opened." << endl;
        return -1;
    }


    process(cap);

    return 0;
}