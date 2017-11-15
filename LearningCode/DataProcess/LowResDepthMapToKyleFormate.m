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
load /afs/cs/group/reconstruction3d/scratch/Min/data/filename.mat
      VertYImg = 2272;
        HoriXImg = 1704;
fx = 2400.2091651084;
fy = 2407.3312729885838;
Ox_de = 1110.7122391785729;%2272/2; %
Oy_de = 833.72104535435108;%1704/2; %
a = 1704/fy; %0.70783777; %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
b= 2272/fx; %0.946584169;%0.085; % vertical physical size of image plane normalized to focal length (in meter)
Ox = 1-Oy_de/1704%0.489272914; % camera origin offset from the image center in horizontal direction
Oy = 1-Ox_de/2272%0.488886982; % camera origin offset from the image center in vertical direction
for i=41:size(filename,2)
    depthfile = strrep(filename{i},'img','depth');
    load(['/afs/cs/group/reconstruction3d/Data/depthMap/' depthfile]);
    depthMap = imresize(depthMap,[55 305],'nearest');
    pixelReservoir(:,6) = depthMap(:);
    [Ix Iy] = meshgrid(((305:-1:1)-0.5)/305*1704,((1:55)-0.5)/55*2272);
    pixelReservoir(:,4) = Iy(:);
    pixelReservoir(:,5) = Ix(:);
    ray = cat(3,a*(1-pixelReservoir(:,5)/HoriXImg-Ox),b*(1-pixelReservoir(:,4)/VertYImg-Oy),ones(size(pixelReservoir,1),1));
    ray = ray./repmat(sqrt(sum(ray(:,:,1).^2+ray(:,:,2).^2+ray(:,:,3).^2,3)),[1 1 3]);
    Position3DTure=im_cr2w_cr(pixelReservoir(:,6),ray); 
    pixelReservoir(:,1:3) = [Position3DTure(2,:)' Position3DTure(1,:)' Position3DTure(3,:)'];

    depthfile = strrep(filename{i},'img','depth_sph_corr');
    save(['/afs/cs/group/reconstruction3d/scratch/Min/laserdata/' depthfile],'pixelReservoir');
end    
