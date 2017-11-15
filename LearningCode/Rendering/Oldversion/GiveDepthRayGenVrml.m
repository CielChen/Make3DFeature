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
function [] = GiveDepthRayGenVrml( depthFolder,learned);

% selected image with low error as train data set
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

load([ScratchDataFolder '/data/MaskGSky.mat']);
file = dir([ScratchDataFolder '/' depthFolder])
NuPics = size(file,1);
for i = 1:NuPics
        if ~any(strfind(file(i).name,'.mat'))
           continue;
        end
        i
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
        
        if learned
           depthfile = strrep(filename{i},'img','depth_learned'); % the depth filename
           load([ScratchDataFolder '/' depthFolder '/' depthfile '.mat']);
           depthMap(depthMap <0) =100;
        else 
           depthfile = strrep(filename{i},'img','depth_sph_corr'); % the depth filename
           load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
           LaserDepth = Position3DGrid(:,:,4);
           clear Position3DGrid;         
        end

        % initalize the ray
        RayPorjectImgMapY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]);
        RayPorjectImgMapX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);
        RayPorjectImgMapY = ((VertYNuDepth+1-RayPorjectImgMapY)-0.5)/VertYNuDepth - Oy;
        RayPorjectImgMapX = (RayPorjectImgMapX-0.5)/HoriXNuDepth - Ox;
        Ray = RayImPosition(RayPorjectImgMapY,RayPorjectImgMapX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        Ray = permute(Ray,[3 1 2]);
        [Position3D] = im_cr2w_cr(depthMap,permute(Ray,[2 3 1]));
        [VrmlName] = vrml_test_faceset_goodSkyBoundary( filename{i}, Position3D, depthMap, permute(Ray,[2 3 1]), depthFolder, ...
                     [],[], 0, maskSky{i}, maskg{i}, 1, 0, a_default, b_default, Ox_default, Oy_default);
        system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
        delete([ScratchDataFolder '/vrml/' VrmlName]);
end
