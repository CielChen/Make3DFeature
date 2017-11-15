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
function []=CheckImage2LaseDepthMap();

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename;
%a= 0.70783777; %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
%b = 0.946584169;%0.085; % vertical physical size of image plane normalized to focal length (in meter)
%Ox = 0.489272914; % camera origin offset from the image center in horizontal direction
%Oy = 0.488886982; % camera origin offset from the image center in vertical direction
HoriXImg = 1704;
VertYImg = 2272;

%GeneralDataFolder = '/afs/cs/group/reconstruction3d/Data';
%ScratchDataFolder = '/afs/cs/group/reconstruction3d/scratch/Min';

%load([ScratchDataFolder '/data/filename.mat']);
NuPics =size(filename,2);
for k = 1:NuPics
    depthfile = strrep(filename{k},'img','depth_sph_corr');
    temp = dir([GeneralDataFolder '/laserdata/' depthfile '.mat']);
    if size(temp,1) ==0
       k
       %load([GeneralDataFolder '/laserdata/' depthfile '.mat']);
       unix(['rm ' GeneralDataFolder '/' ImgFolder '/' filename{k} '.jpg']);
    %pixelReservoir(:,5) = 1704-pixelReservoir(:,5);
    % generate 3d position in camera coordinate
    %ray = cat(3,(a*pixelReservoir(:,5)/HoriXImg-Ox),-(b*pixelReservoir(:,4)/VertYImg-Oy),ones(size(pixelReservoir,1),1));
    %ray = ray./repmat(sqrt(sum(ray(:,:,1).^2+ray(:,:,2).^2+ray(:,:,3).^2,3)),[1 1 3]);
    %Position3DTure=im_cr2w_cr(pixelReservoir(:,6),ray); 
    %pixelReservoir(:,1:3) = Position3DTure';
    %save([ScratchDataFolder '/laserdataMin/' depthfile '.mat'],'pixelReservoir');
       %gen_position3dgrid(depthfile,pixelReservoir,VertYNuDepth,HoriXNuDepth,2272,1704);
    end
end    
