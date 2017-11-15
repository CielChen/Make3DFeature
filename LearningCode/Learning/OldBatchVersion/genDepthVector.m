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
function [DepthVector] = genDepthVector(Depth,RowTop,RowBottom,ColumnLeft,ColumnRight,i);
% this function generate the Deoth Vector respect to the Row

% This function set the feature to the right format of Feature Vector
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename NuRow_default;

NuRow = NuRow_default;

%lodd pics info
PicsinfoName = strrep(filename{i},'img','picsinfo');
temp = dir([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
if size(temp,1) == 0
   a = a_default;
   b = b_default;
   Ox = Ox_default;
   Oy = Oy_default;
   Horizon = Horizon_default;
else
   load([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
end

% generate the range of the row for the same thi (weight value)
%RowskyBottom = floor(NuRow/2);
%PatchSkyBottom = ceil(VertYNuDepth*(1-Horizon));
%if row <= RowskyBottom
%   PatchRowRatio = PatchSkyBottom/RowskyBottom;
%   RowTop = round((row-1)*PatchRowRatio+1);
%   RowBottom = round(row*PatchRowRatio);
%else
%   PatchRowRatio = (VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
%   RowTop = round((row-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
%   RowBottom = round((row-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
%end

DepthVector = [];

% pick out the depth
shift = 9*[0 0];%; -1 0; 1 0; 0 -1; 0 1];
for l = 1:1
    [Ix Iy] = meshgrid(max(min(ColumnLeft+shift(l,1):ColumnRight+shift(l,1),HoriXNuDepth),1),...
                       max(min(RowTop+shift(l,2):RowBottom+shift(l,2),VertYNuDepth),1));
    maskNeibor = sub2ind([VertYNuDepth, HoriXNuDepth], Iy(:), Ix(:));
    DepthVector =[ DepthVector ;Depth(maskNeibor)];
end

return;
