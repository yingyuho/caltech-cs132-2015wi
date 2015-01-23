#include <cstdio>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define CAP_WIDTH 320
#define CAP_HEIGHT 240

int main(int argc, char const *argv[])
{
    Mat camera_matrix;
    Mat dist_coeffs;
    vector< vector<Point2f> > image_points;
    vector< vector<Point3f> > object_points;

    Size image_size(CAP_WIDTH, CAP_HEIGHT);
    Size board_size(9, 6);

    Mat img, img2;
    vector<Point2f> image_corners;
    vector<Point3f> object_corners;

    char filename[256];
    string window_name = "S = Jump to calib. | Q/Esc = Abort";
    int count = 0;
    bool pattern_found;
    bool to_skip = false;
    char key;

    // Model corner coords
    for (int i = 0; i < board_size.height; i++)
        for (int j = 0; j < board_size.width; j++)
            object_corners.push_back(Point3f(i, j, 0.0f));

    namedWindow(window_name, WINDOW_AUTOSIZE);

    for (;;) {
        // Load IMG = IMG_SAVED000.bmp, IMG_SAVED001.bmp, ...
        sprintf(filename, "IMG_SAVED%.3d.bmp", count++);
        img = imread(filename, 0);
        printf("IMG = %s\n", filename);

        // Break if no more images
        if (img.empty())
            break;

        // Find corners
        pattern_found = findChessboardCorners(img, board_size, image_corners);

        if (pattern_found) {
            // Refine corner locations
            cornerSubPix(img, image_corners, 
                Size(11, 11), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            // Store corner coords on images and model
            image_points.push_back(image_corners);
            object_points.push_back(object_corners);
        }

        if (to_skip)
            continue;

        // Show detected corners
        img.copyTo(img2);

        drawChessboardCorners(img2, board_size, image_corners, pattern_found);

        imshow(window_name, img2);

        key = (char) waitKey(400);

        switch (key) {
        // Q or Esc to abort
        case 'q':
        case 'Q':
        case 27: // Esc
            return 0;
        // S to skip displaying images
        case 's':
        case 'S':
            to_skip = true;
            continue;
        default:
            continue;
        };
    }

    // Calibration
    vector<Mat> rvecs, tvecs;
    double rms = calibrateCamera(
        object_points,  // 3D points
        image_points,   // image points
        image_size, 
        camera_matrix,  // camera intrinsic matrix
        dist_coeffs,    // distortion coefficient matrix
        rvecs, tvecs,   // rotation and translation vector
        0);

    printf("RMS = %lf\n", rms);

    // Save parameter to XML file
    FileStorage fs("test.yml", FileStorage::WRITE);
    fs  << "intrinsic" << camera_matrix 
        << "distcoeff" << dist_coeffs;

    return 0;
}