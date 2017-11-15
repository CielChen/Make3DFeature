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
function [OccluList BoundaryLaserOccluHori BoundaryLaserOccluVert]=LaserOccluLabel(k,nList);


global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

BpWidthV = ceil(0.01*VertYNuDepth);
BpWidthH = ceil(0.05*HoriXNuDepth);
SE = strel('rectangle',[BpWidthV BpWidthH]);
ThreVert = 0.5;
ThreHori = 0.5;
ThreFar = 15;
% load data
depthfile = strrep(filename{k},'img','depth_sph_corr'); % the depth filename
load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
LaserDepth = Position3DGrid(:,:,4);

load([ScratchDataFolder '/data/CleanSup/CleanSup' num2str(k) '.mat']);
Sup = double(Sup);
MaxSup = max( Sup(:));
MaskSky = Sup ==0;

BoundaryPVert = conv2(Sup,[1 MaxSup],'valid').*( conv2(Sup,[1 -1],'valid') >0)...
                .* ~MaskSky(:,1:(end-1)) .* ~MaskSky(:,2:end); % two step build up hash index 1) Left > Righ index
BoundaryPVert = BoundaryPVert + conv2(Sup,[MaxSup 1],'valid').* (conv2(Sup,[-1 1],'valid') >0)...
                .* ~MaskSky(:,1:(end-1)) .* ~MaskSky(:,2:end); % 2) Left < Righ index
BoundaryPHori = conv2(Sup,[1; MaxSup],'valid').* (conv2(Sup,[1; -1],'valid') >0)...
                .* ~MaskSky(1:(end-1),:) .* ~MaskSky(2:end,:); % two step build up hash index 1) Top > bottom index
BoundaryPHori = BoundaryPHori + conv2(Sup,[MaxSup; 1],'valid').*( conv2(Sup,[-1; 1],'valid') >0)...
                .* ~MaskSky(1:(end-1),:) .* ~MaskSky(2:end,:); % 2) Top < Bottom index

% detect occlusion in both vertical and horizontal direction
DiffDepthVert = abs(conv2(LaserDepth,[1; -1],'valid'));
DiffDepthHori = abs(conv2(LaserDepth,[1 -1],'valid'));
FraDiffDepthVert = DiffDepthVert./ sqrt(LaserDepth(1:(end-1),:) .* LaserDepth(2:end,:) );
%FraDiffDepthVert = DiffDepthVert./ min(LaserDepth(1:(end-1),:) , LaserDepth(2:end,:) );
OccFraDiffDepthVert = FraDiffDepthVert > ThreVert;
OccFraDiffDepthVert(LaserDepth(1:(end-1),:) > ThreFar & LaserDepth(2:end,:) > ThreFar) = 0;
FraDiffDepthHori = DiffDepthHori./ sqrt( LaserDepth(:,1:(end-1)) .* LaserDepth(:,2:end) );
%FraDiffDepthHori = DiffDepthHori./ min( LaserDepth(:,1:(end-1)) , LaserDepth(:,2:end) );
OccFraDiffDepthHori = FraDiffDepthHori > ThreHori;
OccFraDiffDepthHori(LaserDepth(:,1:(end-1)) > ThreFar & LaserDepth(:,2:end) > ThreFar) = 0;

BoundaryLaserOccluHori = (imdilate(OccFraDiffDepthVert, SE).*(BoundaryPHori ~= 0));
BoundaryLaserOccluVert = (imdilate(OccFraDiffDepthHori, SE).*(BoundaryPVert ~= 0));
OccluMap = zeros(size(LaserDepth));
OccluMap(:,1:(end-1)) = imdilate(OccFraDiffDepthHori, SE);
OccluMap(1:(end-1),:) = imdilate(OccFraDiffDepthVert, SE);
%figure(200);subplot(1,2,1);
%figure;
%disp('plot');
%Img = imresize(imread([GeneralDataFolder '/' ImgFolder '/' filename{k} ],'jpg'), [SegVertYSize SegHoriXSize]);
%Img(:,:,1) = 255*imresize(OccluMap, [ SegVertYSize, SegHoriXSize]);
%image(Img);

OccluHash = setdiff(unique([BoundaryPHori(BoundaryLaserOccluHori ~= 0); BoundaryPVert(BoundaryLaserOccluVert ~= 0)]),0);
List = nList(:,1)*MaxSup + nList(:,2);
OccluList = false*ones(size(List)); 
if isempty(OccluHash)
   return;
end
mask = sum(repmat(List,[1 size(OccluHash,1)]) == repmat(OccluHash',[size(List,1) 1]),2) > 0;
OccluList(mask) = true;
return;
