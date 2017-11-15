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
function []=genVrmlLaserData()

% this function generate the VRML from the laser depthMap
% Use this to see if the depthMap makes sense


% declaim global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename;

% load data
%load([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat']); % superpixel_index

NuPics = size(filename,2);
for i = 10%1 : NuPics
    
    % 1) load picsinfo 
    PicsinfoName = strrep(filename{i},'img','picsinfo');
    temp = dir([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
    if size(temp,1) == 0
        a = a_default;
        b = b_default;
        Ox = Ox_default;
        Oy = Oy_default;
        Horizon = Horizon_default;
        VertYImg = 2272;
        HoriXImg = 1704;
    else    
        load([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
    end    
    
    % 2) Loadin the 3d position and corresponding image position generated
    depthfile = strrep(filename{i},'img','depth_sph_corr'); % the depth filename(without .file extension) associate with the *jpg file
    % from Rajiv's code
    temp = dir([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
    if size(temp,1) ~=0
       i
       load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']); 
%    load([ScratchDataFolder '/laserdata/' depthfile '.mat']);
    %[Position3DGrid] = gen_position3dgrid(depthfile,pixelReservoir,VertYNuDepth,HoriXNuDepth,VertYImg,HoriXImg);
    % ============ changable ================
    %pixelReservoir(:,5) = 1704 - pixelReservoir(:,5);
    %[Position3DGrid] = gen_position3dgrid(depthfile,pixelReservoir,VertYNuDepth,HoriXNuDepth,VertYImg,HoriXImg);
    % =======================================================

    % add superpixel index to the laserdata
 %   sup = LowResImgIndexSuperpixelSep{i,1};
 %   Indices = sub2ind(size(sup),max(min(ceil((pixelReservoir(:,4)/VertYImg)*VertYNuDepth),VertYNuDepth),1),...
 %                     max(min(ceil((1-pixelReservoir(:,5)/HoriXImg)*HoriXNuDepth),HoriXNuDepth),1));
 %   pixelReservoir(:,7) = sup(Indices);
    %save([ScratchDataFolder '/laserdataMin/' depthfile '.mat'],'pixelReservoir');

    % generate specific ray for whole pics
%    RayCorner = GenerateRay(HoriXNuDepth,VertYNuDepth,'corner',a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
    RayCenter = GenerateRay(HoriXNuDepth,VertYNuDepth,'center',a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
    Position3D = im_cr2w_cr(Position3DGrid(:,:,4),RayCenter);
%    size(RayCorner)
%    size(RayCenter)
    
    % generate 3d position in camera coordinate
%    ray = cat(3,a*(1-pixelReservoir(:,5)/HoriXImg-Ox),b*(1-pixelReservoir(:,4)/VertYImg-Oy),ones(size(pixelReservoir,1),1));
%    ray = ray./repmat(sqrt(sum(ray(:,:,1).^2+ray(:,:,2).^2+ray(:,:,3).^2,3)),[1 1 3]);
%    Position3DTure=im_cr2w_cr(pixelReservoir(:,6),ray); 
 
    % 3) calculate each plane parameter for each superpixel
%    [PlaneParameterTure]=fit_all_planes([ pixelReservoir(:,[2 1 3])...
%                         max(min(ceil((pixelReservoir(:,4)/VertYImg)*VertYNuDepth),VertYNuDepth),1) ...
%                         max(min(ceil((1-pixelReservoir(:,5)/HoriXImg)*HoriXNuDepth),HoriXNuDepth),1) ...
%                          pixelReservoir(:,[7])],Position3DGrid(:,:,[2 1 3 4]),...
%                         permute(RayCorner,[3 1 2]),sup); % hard work around 2min
    
 %   DepthTureProjPics = 1./sum(RayCenter.*permute(reshape(PlaneParameterTure(1:3,sup),3,VertYNuDepth,HoriXNuDepth),[2 3 1]),3);
%    Position3DTureProjPics=im_cr2w_cr(DepthTureProjPics,RayCenter);
%    Position3DTureProjPics=im_cr2w_cr(DepthTureProjPics,RayCenter);
%    vrml_test_faceset_triangle(filename{i},Position3DTureProjPics,RayCenter,'fit');
    [VrmlName] = vrml_test_faceset_goodSkyBoundary(filename{i},Position3D,Position3DGrid(:,:,4),RayCenter, 'grid',...
                 [],[],0,zeros(VertYNuDepth,HoriXNuDepth), zeros(VertYNuDepth,HoriXNuDepth),...
                 1,0,a_default,b_default,Ox_default,Oy_default);
    system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
    delete([ScratchDataFolder '/vrml/' VrmlName]);

    %vrml_test_faceset_triangle(filename{i},permute(Position3DGrid(:,:,1:3),[3 1 2]),RayCenter,'real');
    
    % save
%    PlaneParameter{i} = PlaneParameterTure;
%    DepthTureProj{i} = DepthTureProjPics;
%    Position3DTureProj{i} = Position3DTureProjPics;
   end
end

%save([ScratchDataFolder '/data/PlaneParameter.mat'], 'PlaneParameter');
%save([ScratchDataFolder '/data/DepthTureProj.mat'], 'DepthTureProj');
%save([ScratchDataFolder '/data/Position3DTureProj.mat'], 'Position3DTureProj');
return;
