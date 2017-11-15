/*
% *  This code was used in the following articles:
% *  [1] Learning 3-D Scene Structure from a Single Still Image, 
% *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
% *      In ICCV workshop on 3D Representation for Recognition (3dRR-07), 2007.
% *      (best paper)
% *  [2] 3-D Reconstruction from Sparse Views using Monocular Vision, 
% *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
% *      In ICCV workshop on Virtual Representations and Modeling 
% *      of Large-scale environments (VRML), 2007. 
% *  [3] 3-D Depth Reconstruction from a Single Still Image, 
% *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
% *      International Journal of Computer Vision (IJCV), Aug 2007. 
% *  [6] Learning Depth from Single Monocular Images, 
% *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
% *      In Neural Information Processing Systems (NIPS) 18, 2005.
% *
% *  These articles are available at:
% *  http://make3d.stanford.edu/publications
% * 
% *  We request that you cite the papers [1], [3] and [6] in any of
% *  your reports that uses this code. 
% *  Further, if you use the code in image3dstiching/ (multiple image version),
% *  then please cite [2].
% *  
% *  If you use the code in third_party/, then PLEASE CITE and follow the
% *  LICENSE OF THE CORRESPONDING THIRD PARTY CODE.
% *
% *  Finally, this code is for non-commercial use only.  For further 
% *  information and to obtain a copy of the license, see 
% *
% *  http://make3d.stanford.edu/publications/code
% *
% *  Also, the software distributed under the License is distributed on an 
% * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
% *  express or implied.   See the License for the specific language governing 
% *  permissions and limitations under the License.
% *
% */
////////////////////////////////////////////////////////////////////////
//
// COneShot3d.cpp
//
// This is the core View3 OpenCV program. The program reads an
// image from a file, created a wrl file and stores the result. 
//
////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <COneShot3d.h>

#define IN_IMAGE     	argv[1] 
#define IN_ImgPath      	argv[2]
#define IN_OutPutFolder 	argv[3]
#define IN_ScratchFolder	argv[4]
#define IN_taskName	argv[5]
#define IN_DepthPara	argv[6]
#define IN_VarPara		argv[7]
#define IN_GroundSkyPara	argv[8]
#define IN_FeaPara		argv[9]
#define IN_SFeaPara	argv[10]

#if !defined(MAX)
#define MAX(A, B)       ((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define MIN(A, B)       ((A) < (B) ? (A) : (B))
#endif


int main(int argc, char *argv[])
{
  IplImage* img = 0; 
  int height,width,step,channels;
  uchar *data;
  double *MedSup;
  double *Sup;
  int i,j,k;i
  bool evalScratchFlag=0;

  if(argc<2){
    printf("Usage: main <image-file-name>\n\7");
    exit(0);
  }

  // load an image  
  img=cvLoadImage(argv[1]);
  if(!img){
    printf("Could not load image file: %s\n",argv[1]);
    exit(0);
  }

  // get the image data
  height    = img->height;
  width     = img->width;
  step      = img->widthStep;
  channels  = img->nChannels;
  data      = (uchar *)img->imageData;
  printf("Processing a %dx%d image with %d channels\n",height,width,channels); 

  // Set Default parametersi
  if (IN_ScratchFolder == '')
     evalScratchFlag = 1;
  
  void DefaultParaValues::setFolderNames(char *taskName,
                  char *DepthPara,
                  char *VarPara,
                  char *GroundSkyPara,
                  bool ScratchFlag,
                  char *SFeaPara,
                  char *FeaPara) {taskName=IN_taskName;
                                  DepthPara=IN_DepthPara;
                                  VarPara=IN_VarPara;
                                  IN_GroundSkyPara=GroundSkyPara;
                                  ScratchFlag=evalScratchFlag;
                                  SFeaPara=IN_SFeaPara;
                                  FeaPara=IN_FeaPara;}

/*
  // create a window
  cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE); 
  cvMoveWindow("mainWin", 100, 100);

  // show the image
  cvShowImage("mainWin", img );
*/  
  
 /*

 // invert the image
  for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)
    data[i*step+j*channels+k]=255-data[i*step+j*channels+k];
 */

//% ***************************************************

//% Features ===========================================

//  % 1) Basic Superpixel generation and Sup clean
    Cgen_Sup_efficient(MedSup, Sup, DefaultParaValues, img);

//  % 2) Texture Features and inner multiple Sups generation

//  % 3) Superpixel Features generation


//% Inference ==========================================

//  % 1) Generate Ground and Sky mask

//  % 2) Clean Sup{1} (1st Scale) according to the sky mask

//  % 3) Generate predicted (depth:1 Variance:2 )

//  % 4) Plane Parameter MRF

//% ***************************************************


  // wait for a key
  cvWaitKey(0);

  // release the image
  cvReleaseImage(&img );
  return 0;
}
