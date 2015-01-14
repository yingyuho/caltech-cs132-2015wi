#include <string>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    const string img_file_name = "First_Image.bmp";
    Mat img;
    // Read the image
    img = imread(img_file_name, 1);
    // Create a window
    namedWindow(img_file_name, WINDOW_AUTOSIZE);
    // Show the image
    imshow(img_file_name, img);
    // Wait for any key
    waitKey(0);
    // Destroy the window
    destroyWindow(img_file_name);
    
    return 0;
}