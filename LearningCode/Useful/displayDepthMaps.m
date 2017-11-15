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
function displayDepthMaps(depthMap1, depthMap2, depthMap3, depthMap4, depthMap5, depthMap6, n, trainOrTest, mode)
% this function displays the _Real_ values of depthmaps in the display, so
% that you can compare them side-by-side.

global minDistance maxDistance
minDistance = min(min(depthMap1));
maxDistance = max(max(depthMap1));

if nargin < 9,  mode = 0;       end
if nargin < 8,  trainOrTest = 1,end
if nargin < 7,  n = 0;          end
if nargin < 6,  depthMap6=0;    end
if nargin < 5,  depthMap5=0;    end
if nargin < 4,  depthMap4=0;    end
if nargin < 3,  depthMap3=0;    end
if nargin < 2,  depthMap2=0;    end
if nargin < 1,  disp('Why the **ck you called me: No Image to display.');     return;    end

if n >= 0
    jetMap = jet;
    jetMap = 1-jetMap(end:-1:1,:);
    jetMap = jetMap(end:-1:1,:);

    jetMap = imresize(jetMap, [256 3], 'bilinear');

    displayNorm = size(jetMap,1);

    warning off;
    depthMap1 = uint8( displayNorm* (depthMap1 - minDistance) / (maxDistance - minDistance) );
    depthMap2 = uint8( displayNorm* (depthMap2 - minDistance) / (maxDistance - minDistance) );
    depthMap3 = uint8( displayNorm* (depthMap3 - minDistance) / (maxDistance - minDistance) );
    depthMap4 = uint8( displayNorm* (depthMap4 - minDistance) / (maxDistance - minDistance) );
    depthMap5 = uint8( displayNorm* (depthMap5 - minDistance) / (maxDistance - minDistance) );
    depthMap6 = uint8( displayNorm* (depthMap6 - minDistance) / (maxDistance - minDistance) );
    warning on;

    %figure, hist( double(depthMap1(:) )) ;
    %figure, hist( double(depthMap2(:) )) ;

    [max( depthMap1(:) ), min( depthMap1(:) )];

    if mode == 0,   figure, end
    if mode, figure, else, subplot(1,2,1), end
    image( depthMap1 );  colormap( jetMap );    axis off;   axis tight;%colorbar
     if mode, figure, else, subplot(1,2,2), end
     image( depthMap2 );  colormap( jetMap );    axis off;   axis tight;%colorbar
%     if mode, figure, else, subplot(1,4,3), end 
%     image( depthMap3 );  colormap( jetMap );    axis off;   axis tight;%colorbar
%     if mode, figure, else, subplot(1,4,4), end
%     image( depthMap4 );  colormap( jetMap );    axis off;   axis tight;%colorbar
%     if mode, figure, else, subplot(2,3,5), end
%     image( depthMap5 );  colormap( jetMap );    axis off;   axis tight;%colorbar
%     if mode, figure, else, subplot(2,3,6), end
%     image( depthMap6 );  colormap( jetMap );    axis off;   axis tight;%colorbar
end

if n ~= 0
%    if n > 0, figure(1000*n), 
%    else, figure(-1000*n);
%    if n > 0, figure(100), 
%    else, figure(-100);
%    end
    n = abs(n);
    global imageDirectory
    dirList = dir([imageDirectory '*right*.jpg']);
    if trainOrTest == 1
        actualImageNumber = floor((n-1)*4/3+1);
    else
        actualImageNumber = n*4;
    end
    imageFilename = dirList(actualImageNumber).name;
    A = imread([imageDirectory imageFilename]);
    imshow(A);
end
return;
