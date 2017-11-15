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
function [seglist ]=edgeSegDetection(Img,k,EdgeReady)

displayFlag = false;

% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

%if nargin < 1
%Img = imread('/afs/cs.stanford.edu/group/reconstruction3d/Data/TestImg/img-combined1-p-108t0.jpg');
% Img = imread('/afs/cs/group/reconstruction3d/Data/moreInternetImages/4.jpg');
%end

% Img = imresize(Img, 0.5, 'nearest');

% this function detect the edge segment
% output the start position and end position of each line segment
[x y dummy] = size(Img);

if ~ EdgeReady
   ImgYCbCr = rgb2ycbcr(Img);
   %method = {'sobel','Prewitt','Roberts','zerocross','Canny','log'};
   method = {'log'}; % for plane Parameter MRF
%    method = {'sobel'};
   NuMethod = size(method,2);
   edgeImgTotal = zeros(x, y);
   for i = 1: NuMethod
      [edgeImg, thresh] = edge(ImgYCbCr(:,:,1),method{i},[ ], 4);
%       [edgeImg, thresh] = edge(ImgYCbCr(:,:,1),method{i},[ ], 'both');
      edgeImg = edge(ImgYCbCr(:,:,1),method{i},thresh/4, 4);
%       edgeImg = edge(ImgYCbCr(:,:,1),method{i},thresh/4, 'both');
      edgeImgTotal = edgeImgTotal | edgeImg;
   end   
else
   edgeImgTotal = Img;
end
% 
 [edgelineList edgeline] = edgelink(edgeImgTotal,10); % hardwork
% [edgelineList edgeline] = edgelink(edgeImgTotal,5); % hardwork
tol = 3;
angtol=0.01;  % .01 for plane Parameter MRF
linkrad = 3;
%minLineLength = 45;
minLineLength = 20;
[seglist, nedgeIm] = lineseg(edgelineList,tol,angtol,linkrad, minLineLength); % hardwork

if displayFlag,
 figure(100), image(Img);
hold on; drawseg(seglist,100,2);
%saveas(100,[ScratchDataFolder '/LineSeg/' filename{k} '_lineSeg.jpg']);
end
