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
function [Bp, OccluMap, BoundaryLaserOccluHori, BoundaryLaserOccluVert] = ...
          LaserDetectOcc2Bp( LaserDepth, BpWidthV, BpWidthH, BoundaryPVert, BoundaryPHori, Hash2Ind, dist)

% This function assign 1 to the bounday labeled as occlusion with a probability distribution, 
% otherwise -1;
if nargin <7
   dist = 'uniform';
end

% inital parameters
ThreVert = 0.5;
ThreHori = 0.5;
ThreFar = 15;
SE = strel('rectangle',[BpWidthV BpWidthH]);


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

% form the OccluMap and BoundaryLaserOccluHori, BoundaryLaserOccluVert
BoundaryLaserOccluHori = imdilate(OccFraDiffDepthVert, SE).*(BoundaryPHori ~=0 );
BoundaryLaserOccluVert = imdilate(OccFraDiffDepthHori, SE).*(BoundaryPVert ~=0 );
OccluMap = zeros(size(LaserDepth));
if ~strcmp(dist,'uniform')
    %assume norm dis
     
end
   OccluMap(1:(end-1),:) = BoundaryLaserOccluHori;
   OccluMap(2:(end),:) = BoundaryLaserOccluHori;
   OccluMap(:,1:(end-1)) = BoundaryLaserOccluVert;
   OccluMap(:,2:(end)) = BoundaryLaserOccluVert;

% specifiy the label in Bp
Boundary = setdiff(unique([BoundaryPVert( (BoundaryLaserOccluVert ~= 0)); BoundaryPHori( (BoundaryLaserOccluHori ~= 0))]), 0);
Bp = -ones( sum(Hash2Ind ~=0),1);
Bp(Hash2Ind(Boundary)) = 1;
% Bp = -ones( sum(Hash2Ind ~=0),1);
% for i = Boundary' 
%    maskH = BoundaryPVert == i;
%    maskV = BoundaryPHori == i;
% %    Bp(Hash2Ind(i)) = mean( [BoundaryLaserOccluVert(maskV); BoundaryLaserOccluHori(maskH) ]);
%    Bp(Hash2Ind(i)) = any( [BoundaryLaserOccluVert(maskV); BoundaryLaserOccluHori(maskH) ] == 1);
% end

return;
