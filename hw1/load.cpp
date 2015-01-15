#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    const string img_file_name = "First_Image.bmp";
    Mat img;
    // Read the image
    img = imread(img_file_name, CV_LOAD_IMAGE_COLOR);
    // Create a window
    namedWindow(img_file_name, WINDOW_AUTOSIZE);
    // Show the image
    imshow(img_file_name, img);
    // Wait for any key
    waitKey(0);

    return 0;
}