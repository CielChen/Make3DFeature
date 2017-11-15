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

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include "mex.h"

/* Input Arguments */
#define COORD3D_IN      prhs[0]
#define COORDIm_IN      prhs[1]
#define SUP             prhs[2]
#define IMG_NAME        prhs[3]
#define WRL_NAME        prhs[4]
#define WRL_PATH        prhs[5]
#define GRID            prhs[6]
#define ATTACH            prhs[7]

/* OutPUT Arguments */
//#define WRL_NAME        plhs[0]

/* Definition */
#if !defined(BOU)
#define BOU(A, B, C)       ((A) > (B) ? (C) : (A))
#endif
#if !defined(CheckMinus)
#define CheckMinus(A, B)       ((A) == (-1) ? (B) : (A))
#endif

void WriteVrml( double ThreeDCoord[],
                double ImCorrd[],
                double Sup[],
                char *ImgName,
                char *wrlName,
                char *WrlFolder,
                char *WrlName,
                unsigned int vr,
                unsigned int hr,
                int grid,
                int attach)
{
   printf("grid %d",grid);
   printf("attach %d",attach);
 
   int displayFlag = 0;

 WrlName = "hi.wrl"; 
/* int *BoundaryPt= new int(vr*hr);*/
 int MaxV = (vr*hr); 
 int *Index = new int[MaxV];
 for (int i = 0; i < MaxV; i++){//Min Tried
	Index[i] = -2; //Initialize to a unique value
 }
 int *ValidVertices= new int[MaxV];
 int CPtr; //Index of the 2-D vr x hr map
 
 /* Identify the Boundary Point of each Superpixel*/
 int IndCount = 0; /* Index for wrl start from 0*/
 for (int k = 0; k < hr; k++){
     CPtr = vr*k;
     int CPtrMark = CPtr;
     for (int i = 0; i < vr; i++){
         int center = (int) Sup[CPtr];
         if (center == 0 && i != (vr -1)){ // don't assign index to sky except the lowest row
		Index[CPtr] = -1;//Min TRied
         }
         else if (
             k != 0 && k != (hr-1) && /*excluding the boundary points*/
             i != 0 && i != (vr-1) && /*excluding the boundary points*/
             (int) Sup[ CPtr + 1] == center &&
             (int) Sup[ CPtr - 1] == center &&
             (int) Sup[ CPtr + vr] == center &&
             (int) Sup[ CPtr - vr] == center){ //if boundary point down right, or 4 side same supindex, do not assign index
         }
         else{
            int mark;
            if (k != 0 && k != (hr-1) &&
                i != 0 && // not for the top row
                (int) Sup[ CPtr + vr] == center &&
                (int) Sup[ CPtr - vr] == center){ // check left and right sup index if the same 
                mark = Index[ CPtr -vr];
            }
            else{
                mark = IndCount;
                ValidVertices[IndCount] = CPtr;
                IndCount++;
            }

            /* start assigning Index*/
            for ( int j = CPtrMark; j <= CPtr; j++){
		if (Index[j] == -2){ //Min Tried
                   Index[j] = mark;
  		}
//                 printf("mark %d to %d",j,Index[j]); //print out index matrix element //Ashu
                   CPtrMark++;
            }
//              printf("ValidVertices %d",CPtr);
         }
//          printf("CPtr %d \n",CPtr);
         CPtr++;
     }
 }

 /* test Index */
/*   printf("\n");
 for (int j = 0; j < CPtr; j++){
   printf("Inde %d to %d",j,Index[j]); //should print out the same number as "mark" //Ashu
 }*/
 
   if( displayFlag){
	printf("Nu Index %d\n",CPtr);
   }

 
 /* Draw Triangles */
 FILE * fp;
// fp = fopen( "RowColReducTri.wrl", "w"); 
 char fullname[500000]; // big enought
 fullname[0] = '\0';
 strcat( fullname, WrlFolder ); 
 strcat( fullname, wrlName ); 
 strcat( fullname, ".wrl" );

  if( displayFlag ) { 
      printf("fullname = %s",fullname);
  }
 if (!attach){
    fp = fopen( fullname , "w"); 
    fprintf(fp, "#VRML V2.0 utf8\n");

    /* % add navigate_info*/
    fprintf(fp, "NavigationInfo {\n");
    fprintf(fp, "  headlight TRUE\n");
    fprintf(fp, "  type [\"FLY\", \"ANY\"]}\n\n");

    /*% add viewpoint*/
    fprintf(fp, "Viewpoint {\n");
    fprintf(fp, "    position        0 0.0 0.0\n");
    fprintf(fp, "    orientation     0 0 0 0\n");
    fprintf(fp, "    fieldOfView     0.7\n");
    fprintf(fp, "    description \"Original\"}\n");

    /*%============== add background color======*/
    fprintf(fp, "DEF Back1 Background {\n");
    fprintf(fp, "groundColor [.3 .29 .27]\n");
    fprintf(fp, "skyColor [0.31 0.54 0.76]}\n");
    /*%=========================================*/
 }else{
    printf("a+");
    fp = fopen( fullname , "a+"); 
 }

 

 /*% add Shape for texture faceset*/
 fprintf(fp, "Shape{\n");
 fprintf(fp, "  appearance Appearance {\n");
 fprintf(fp, "   texture ImageTexture { url \"./%s.jpg\" }\n",ImgName);
 fprintf(fp, "  }\n");
 fprintf(fp, "  geometry IndexedFaceSet {\n");
 fprintf(fp, "    coord Coordinate {\n");

 /*% insert coordinate in 3d*/
 /*% =======================*/
 fprintf(fp, "      point [ \n");

 for (int i = 0; i < IndCount; i++){
     fprintf(fp, "        %.2f %.2f %.2f,\n",
             ThreeDCoord[ ValidVertices[i]],
             ThreeDCoord[ ValidVertices[i] + MaxV],
             ThreeDCoord[ ValidVertices[i] + MaxV + MaxV]
              );
 }
 fprintf(fp, "      ]\n");
 fprintf(fp, "    }\n");
 
 /*% insert coordinate index in 3d*/
 fprintf(fp, "    coordIndex [\n");

 for (int k = 0; k < (hr-1); k++){
     int CPtr = vr*k;
     for (int i = 0; i < (vr-1); i++){
         if ( Sup[CPtr] == Sup[CPtr + vr + 1]){

            /* Left Low Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr + 1] == Index[CPtr+1+vr] ){ //if Index repeted in the triangle, do nothing
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1] != -1 &&
		    Index[CPtr+1+vr] != -1){//Min Tried
	               fprintf(fp, "              %d %d %d -1,\n",
        	               Index[CPtr],
                	       Index[CPtr+1],
                  		Index[CPtr+1+vr]
	                       );
		}
              }

            /* Right Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr]){
//                 continue;
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1+vr] != -1 &&
		    Index[CPtr+vr] != -1){ //Min Tried
	               fprintf(fp, "              %d %d %d -1,\n",
        	               Index[CPtr],
                	       Index[CPtr+1+vr],
                       	       Index[CPtr+vr]
                       );
		}
            }


         }
         else{ //Sup[CPtr] != Sup[CPtr + vr + 1]
   
            /* Left Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+vr] ||
                 Index[CPtr + 1] == Index[CPtr+vr] ){
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1] != -1 &&
		    Index[CPtr+vr] != -1){ //Min Tried
			fprintf(fp, "              %d %d %d -1,\n",
                          	Index[CPtr],
  	                        Index[CPtr+1],
         		        Index[CPtr+vr]
                        );
		}
            }

            /* Right Low Triangle*/
            if ( Index[CPtr+1] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1]){
//                 continue;
            }
            else{
		if (Index[CPtr+1] != -1 &&
		    Index[CPtr+1+vr] != -1 &&
		    Index[CPtr+vr] != -1){ //Min Tried
               		fprintf(fp, "              %d %d %d -1,\n",
                       		Index[CPtr+1],
                       		Index[CPtr+1+vr],
                       		Index[CPtr+vr]
                        );
		}
            }
         }
         CPtr++;
     }
 }
 fprintf(fp, "    ]\n");

 /*% insert texture coordinate*/
 fprintf(fp, "    texCoord TextureCoordinate {\n");
 fprintf(fp, "      point [\n");
 
for (int i = 0; i < IndCount; i++){
     fprintf(fp, "              %.4g %.4g,\n",
             ImCorrd[ ValidVertices[i]],
             ImCorrd[ ValidVertices[i] + MaxV]
              );
 }
 fprintf(fp, "        ]\n");
 fprintf(fp, "    }\n");
 fprintf(fp, "    texCoordIndex [\n");

 int Tri =0;
 for (int k = 0; k < (hr-1); k++){
     int CPtr = vr*k;
     for (int i = 0; i < (vr-1); i++){
         if ( Sup[CPtr] == Sup[CPtr + vr + 1]){

            /* Left Low Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr + 1] == Index[CPtr+1+vr] ){
//                 continue;
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1] != -1 &&
		    Index[CPtr+1+vr] != -1){//Min Tried
               		fprintf(fp, "              %d %d %d -1,\n",
                       		Index[CPtr],
                       		Index[CPtr+1],
                       		Index[CPtr+1+vr]
                        );
                     	Tri++;
		}
            }
// 
            /* Right Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr]){
//                 continue;
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1+vr] != -1&&
		    Index[CPtr+vr] != -1){ //Min Tried
               		fprintf(fp, "              %d %d %d -1,\n",
                       		Index[CPtr],
                       		Index[CPtr+1+vr],
                       		Index[CPtr+vr]
                        );
               		Tri++;    
		}           
            }

         }
         else{
   
            /* Left Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+vr] ||
                 Index[CPtr + 1] == Index[CPtr+vr] ){
//                 continue;
            }
            else{
		if (Index[CPtr] != -1 &&
		    Index[CPtr+1] != -1 &&
		    Index[CPtr+vr] != -1){ //Min Tried
   	         	fprintf(fp, "              %d %d %d -1,\n",
                       		Index[CPtr],
                       		Index[CPtr+1],
                       		Index[CPtr+vr]
                       	);
               		Tri++;     
		}          
            }

            /* Right Low Triangle*/
            if ( Index[CPtr+1] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1]){
//                 continue;
            }
            else{
		if (Index[CPtr+1] != -1 &&
		    Index[CPtr+1+vr] != -1 &&
		    Index[CPtr+vr] != -1){ //Min Tried
               		fprintf(fp, "              %d %d %d -1,\n",
                       		Index[CPtr+1],
                       		Index[CPtr+1+vr],
                       		Index[CPtr+vr]
                       	);
               		Tri++;     	
		}          
            }
         } //if ( Sup[CPtr] == Sup[CPtr + vr + 1])
         CPtr++;
     }
 }
 fprintf(fp, "    ]\n");
 fprintf(fp, "  }\n");
 fprintf(fp, "}\n");

 /*write Sup*/
 fprintf(fp, "#Sup [\n");
 for (int i = 0; i < IndCount; i++){
     fprintf(fp, "# %d,\n",
             (int)Sup[ ValidVertices[i]]             
              );
 }

 /* ==============drawing the grid ===================*/
 if (grid) {
    fprintf(fp, "Shape{\n");
    fprintf(fp, "  appearance Appearance { material Material {emissiveColor 1 0 0  }}\n");
    fprintf(fp, "    geometry IndexedLineSet {\n");
    fprintf(fp, "    coord Coordinate {\n");
    fprintf(fp, "      point [ \n");
  
    /*writing the 3D coordinate again*/
    for (int i = 0; i < IndCount; i++){
        fprintf(fp, "        %.2f %.2f %.2f,\n",
        ThreeDCoord[ ValidVertices[i]],
        ThreeDCoord[ ValidVertices[i] + MaxV],
        ThreeDCoord[ ValidVertices[i] + MaxV + MaxV]
        );
    }
    fprintf(fp, "      ]\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    coordIndex [\n");

 /* write the index for the grid again*/
 for (int k = 0; k < (hr-1); k++){
     int CPtr = vr*k;
     for (int i = 0; i < (vr-1); i++){
         if ( Sup[CPtr] == Sup[CPtr + vr + 1]){

            /* Left Low Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr + 1] == Index[CPtr+1+vr] ){
            }
            else{
               fprintf(fp, "              %d %d %d -1,\n",
                       Index[CPtr],
                       Index[CPtr+1],
                       Index[CPtr+1+vr]
                       );
              }

            /* Right Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr]){
//                 continue;
            }
            else{
               fprintf(fp, "              %d %d %d -1,\n",
                       Index[CPtr],
                       Index[CPtr+1+vr],
                       Index[CPtr+vr]
                       );
            }

         }
         else{
   
            /* Left Up Triangle*/
            if ( Index[CPtr] == Index[CPtr+1] || 
                 Index[CPtr] == Index[CPtr+vr] ||
                 Index[CPtr + 1] == Index[CPtr+vr] ){
            }
            else{
               fprintf(fp, "              %d %d %d -1,\n",
                       Index[CPtr],
                       Index[CPtr+1],
                       Index[CPtr+vr]
                       );
            }

            /* Right Low Triangle*/
            if ( Index[CPtr+1] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1+vr] ||
                 Index[CPtr+vr] == Index[CPtr+1]){
//                 continue;
            }
            else{
               fprintf(fp, "              %d %d %d -1,\n",
                       Index[CPtr+1],
                       Index[CPtr+1+vr],
                       Index[CPtr+vr]
                       );
            }
         }
         CPtr++;
     }
 }
 /* =====================================================*/

    fprintf(fp, "    ]\n");
    fprintf(fp, "    colorPerVertex FALSE\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
 }

 
 fclose(fp);
 printf("            In WRL, vertices=%d triangles=%d\n",IndCount,Tri);
 delete [] Index;
 delete [] ValidVertices;
 return;
}
/* mexFunctin (matlab interface will be removed later) */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )

{
    double *ThreeDCoord;
    double *ImCorrd;
    double *Sup;
    double *grid;
    double *attach;
    char   *ImgName;
    char   *wrlName;
    char   *WrlFolder;
    char   *WrlName;

    unsigned int v,h,d,vr,hr;
 
   /* ****** WrlFacestHroiReduce( 3DPosition, ImgPosition, SupIndex, gridFlag, Attachflag)**** */

    /* Check for proper number of arguments */

    if (nrhs != 8) {
        mexErrMsgTxt("Eight input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    /* Check the dimensions of the inputs. */

    vr  = (mxGetDimensions(COORD3D_IN))[0];
    hr  = (mxGetDimensions(COORD3D_IN))[1];
    d  = (mxGetDimensions(COORD3D_IN))[2];
    if (!mxIsDouble(COORD3D_IN) || mxIsComplex(COORD3D_IN) ||
        ( d != 3)) {
        mexErrMsgTxt("The depth of the 3DCoord Matrix should be 3.");
    }
    v  = (mxGetDimensions(COORDIm_IN))[0];
    h  = (mxGetDimensions(COORDIm_IN))[1];
    d  = (mxGetDimensions(COORDIm_IN))[2];
    if (!mxIsDouble(COORDIm_IN) || mxIsComplex(COORDIm_IN) ||
        ( d != 2) || ( v != vr) || (h != hr)) {
        mexErrMsgTxt("The depth of the Image Coord Matrix should be 2.");
    }
    v = mxGetM(SUP);
    h = mxGetN(SUP);
    if (!mxIsDouble(SUP) || mxIsComplex(SUP) ||
        (v != vr) || (h != hr)) {
        mexErrMsgTxt("Superpixel is not having the same v and h with others.");
    }
    if (!mxIsChar(IMG_NAME) || mxIsComplex(IMG_NAME) ) {
        mexErrMsgTxt("Image name is not a string.");
    }
    if (!mxIsChar(WRL_NAME) || mxIsComplex(WRL_NAME) ) {
        mexErrMsgTxt("Image name is not a string.");
    }
    if (!mxIsChar(WRL_PATH) || mxIsComplex(WRL_PATH) ) {
        mexErrMsgTxt("WRL path is not a string.");
    }
    v = mxGetM(GRID);
    h = mxGetN(GRID);
    if (!mxIsDouble(GRID) || mxIsComplex(GRID) ||
        (v != 1) || (h != 1)) {
        mexErrMsgTxt("GRID is not scalar.");
    }
    v = mxGetM(ATTACH);
    h = mxGetN(ATTACH);
    if (!mxIsDouble(ATTACH) || mxIsComplex(ATTACH) ||
        (v != 1) || (h != 1)) {
        mexErrMsgTxt("ATTACH is not scalar.");
    }

    /* Create a matrix for the return argument */
    /*WRL_NAME = mxCreateCharMatrixFromStrings(1, mxREAL);
    mxCreateCharArray(mwSize ndim, );*/

    /* Assign pointers to the various parameters */
    //WRL_NAME = mxCreateString(WrlName);

    ThreeDCoord = mxGetPr(COORD3D_IN);
    ImCorrd = mxGetPr(COORDIm_IN);
    Sup =  mxGetPr(SUP);
    ImgName = mxArrayToString(IMG_NAME);
    wrlName = mxArrayToString(WRL_NAME);
    WrlFolder = mxArrayToString(WRL_PATH);
    grid =  mxGetPr(GRID);
    attach =  mxGetPr(ATTACH);
//    printf("WrlFolder %s",WrlFolder);

    /* Do the actual computations in a subroutine */
/*    WriteVrml( ThreeDCoord, ImCorrd, Sup, ImgName, WrlFolder, WrlName, vr, hr);*/
    WriteVrml( ThreeDCoord, ImCorrd, Sup, ImgName, wrlName, WrlFolder, WrlName,
               vr, hr, (int)*grid, (int)*attach);
    return;

}



