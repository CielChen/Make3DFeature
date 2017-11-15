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
function [NonedgeX NonedgeY] = gen_2ndSmooth(sup);
% This is the function that generate the InteractivePermutation matrix for
% the modal of MRF

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

if nargin <1
   sup = ones(VertYNuDepth,HoriXNuDepth);
end

NuPatch = VertYNuDepth*HoriXNuDepth;
NuIndex = setdiff(unique(sup(:)),[0 -1 -2]);
NuSup = size(NuIndex,1);

% generate beta roughly according to the superpixel 
% ex: at the edge of the superpixel beta set to 0 
NonedgeY = zeros(VertYNuDepth, HoriXNuDepth);
NonedgeX = zeros(VertYNuDepth, HoriXNuDepth);

 for i = NuIndex 
     mask = sup==i;
     NonedgeY = NonedgeY | (conv2(double(mask),[1;1;1],'same'))==3;
     NonedgeX = NonedgeX | (conv2(double(mask),[1 1 1],'same'))==3;
 end    
NonedgeY([1 end],:) = 0;
NonedgeX(:,[1 end]) = 0;
NonedgeY = (NonedgeY(:));
NonedgeX = (NonedgeX(:));

% generate Qx Qy
%Qy = spdiags([ones(NuPatch-2,1) -2*ones(NuPatch-2,1) ones(NuPatch-2,1)],[0  1 2],NuPatch-2,NuPatch);
%Qx = spdiags([ones(NuPatch-2,1) -2*ones(NuPatch-2,1) ones(NuPatch-2,1)],[0  VertYNuDepth 2*VertYNuDepth],NuPatch-2,NuPatch);
%Qy = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],[-1 0 1],NuPatch,NuPatch);
%Qx = spdiags([-2*ones(NuPatch,1) ones(NuPatch,1) ones(NuPatch,1)],[0 -VertYNuDepth VertYNuDepth],NuPatch,NuPatch);
%Qx = sparse(reshape(permute(reshape(full(Qy'),VertYNuDepth,HoriXNuDepth,[]),[2 1 3]),NuPatch,[]));

% generate Rx Ry Rz
% Ray = reshape(Ray,NuPatch,1,3);
% RxY = Ray(:,1,1);
% RyY = Ray(:,1,2);
% RzY = Ray(:,1,3);
% RxX = Ray(:,1,1);
% RyX = Ray(:,1,2);
% RzX = Ray(:,1,3);
%Rx = spdiags(Ray(:,1,1),1,NuPatch,NuPatch);
%Ry = spdiags(Ray(:,1,2),1,NuPatch,NuPatch);
%Rz = spdiags(Ray(:,1,3),1,NuPatch,NuPatch);

% IP
%IP =[Qy;Qx];
%IPy = Qy(NonedgeY,:);
%IPx = Qx(NonedgeX,:); 
%IP = [repmat(NonedgeY,[1 NuPatch]).*(Qy*Rx); repmat(NonedgeY,[1 NuPatch]).*(Qy*Ry); repmat(NonedgeY,[1 NuPatch]).*(Qy*Rz);...
%      repmat(NonedgeX,[1 NuPatch]).*(Qx*Rx); repmat(NonedgeX,[1
%      NuPatch]).*(Qx*Ry); repmat(NonedgeX,[1 NuPatch]).*(Qx*Rz)];
return;
