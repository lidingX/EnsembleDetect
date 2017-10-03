#include "utils.h"
#include "OnetoManyMatch.h"
#include "opencv2/imgproc/imgproc.hpp" 
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{ 
	char * targetName = "images/targets/weixin2.png";
	char * templateName = "images/templates/weixin27.png";
	//char * targetName = "images/templates/weixin26.png";
	//char * targetName = "images/targets/weixin.png";
	//char * templateName = "images/templates/weixin1.pn  g";
	//char * targetName = "images/targets/corrected-1 (5).jpg";
	//char * templateName = "images/templates/corrected-2 (5) test4.jpg";
	//char * targetName = "images/targets/corrected-2 (5).jpg";
	//char * templateName = "images/templates/humant.jpg";
	//char * targetName = "images/targets/human.jpg";
	//char * templateName = "images/templates/corrected-3 (7) t.jpg";
	//char * targetName = "images/targets/corrected-3 (7).jpg";
	//char * templateName = "images/templates/corrected-7 (3)t.jpg";
	//char * targetName = "images/targets/corrected-7 (3).jpg";
	//printf("read: %s & %s\n", templateName, targetName);
	Mat templateImage = imread(templateName);
	Mat targetImage = imread(targetName);
	OnetoManyMatch(templateImage, targetImage);
	return 0;
}