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
function [] = extractdata();

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

%SearchFolder{1} = 'MinTest';
%SearchFolder{2} = 'Min96Int';
%SearchFolder{3} = 'SUCampus2';
SearchFolder{1} = 'AshOldTest';
SearchFolder{2} = 'AugTest';
SearchFolder{3} = 'MinDataOct21';
SearchFolder{4} = 'AshOldNewAdd';
%chFolder{1} = 'MinT';
NuFolder = 4;
BatchSize = 10;

for j = 1:NuFolder
%    if ~isempty(dir([ScratchDataFolder '/../' SearchFolder{j} '/data/TextLowResImgIndexSuperpixelSep.mat']))
%       Text(j) = true;
%    else
%       Text(j) = false;
%       disp('Texture false');
%       return;
%    end
%    if ~isempty(dir([ScratchDataFolder '/../' SearchFolder{j} '/Gridlaserdata']))
%       Laserdepth(j) = true;
%    end
    if ~isempty(dir([ScratchDataFolder '/../' SearchFolder{j} '/data/feature_Abs_Whole*']))
       Features(j) = true;
    else
       Features(j) = false;
       disp('feature false');
       return;
    end
end
filename_ori= filename;
NuPics = size(filename_ori,2)
for i = 1: NuPics
    for j = 1 :NuFolder
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/filename.mat']);
        foundFlag = 0;
        TargetNuPics = size(filename,2);
        for k = 1:TargetNuPics
            if strcmp(filename_ori{i}, filename{k})
               filename_ori{i}
               filename{k}
               foundFlag = 1;
               break;
            end
        end
        if foundFlag
           break;
        end
    end
    if foundFlag == 0
         disp('not found');
         return;
    end
    j
    % find k the corresponding batch BaInd and Ind in size the Batch
    BaInd = ceil(k/BatchSize);
    Ind = k - BatchSize*(BaInd-1);
   if Features(j)
        ff = dir([ScratchDataFolder '/../' SearchFolder{j} '/data/feature_Abs_Whole' num2str(BaInd) '_*']);
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/' ff(1).name]);
   end
        % load everything we need
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/LowResImgIndexSuperpixelSep.mat']);
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/DiffLowResImgIndexSuperpixelSep.mat']);
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/MedSeg/MediResImgIndexSuperpixelSep' num2str(k) '.mat']);
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/MaskGSky.mat']);
        load([ScratchDataFolder '/../' SearchFolder{j} '/data/FeatureSuperpixel.mat']);
%        depthfile = strrep(filename_ori{i},'img','depth_learned');
%        load([ScratchDataFolder '/../' SearchFolder{j} '/_LearnDNonsky_depth/' depthfile '.mat']);
%if Laserdepth(j)
%   depthfile = strrep(filename_ori{i},'img','depth_sph_corr');
%   system(['cp ' ScratchDataFolder '/../' SearchFolder{j} '/Gridlaserdata/' depthfile '.mat ' ...
%            ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
%end

    GroupLowResImgIndexSuperpixelSep{i,:} = LowResImgIndexSuperpixelSep{k};
    GroupDiffLowResImgIndexSuperpixelSep(i,:) = DiffLowResImgIndexSuperpixelSep(k,:);
%if Text(j)
%        load([ScratchDataFolder '/../' SearchFolder{j} '/data/TextLowResImgIndexSuperpixelSep.mat']);
%    GroupTextLowResImgIndexSuperpixelSep(i,:,:) = TextLowResImgIndexSuperpixelSep(k,:,:);
%end
if Features(j)
    i
    i - BatchSize*(ceil(i/BatchSize)-1)
    Groupf{i - BatchSize*(ceil(i/BatchSize)-1)} = f{Ind};
    whos('Groupf')
    clear f;
end
    GroupmaskSky{i} = maskSky{k};
    Groupmaskg{i} = maskg{k};
    GroupFeatureSuperpixel{i} = FeatureSuperpixel{k};
    
    % save Medi and depthMap first
    save([ScratchDataFolder '/data/MedSeg/MediResImgIndexSuperpixelSep' num2str(i) '.mat' ],'MediResImgIndexSuperpixelSep');
%    save([ScratchDataFolder '/_LearnDNonsky_depth/' depthfile '.mat'],'depthMap');
if Features(j) && ~rem(i,BatchSize)
   f = Groupf;
   clear Groupf;
   whos('f')
   save([ScratchDataFolder '/data/feature_Abs_Whole' num2str(ceil(i/BatchSize)) '.mat' ],'f');
   clear f;
end

LowResImgIndexSuperpixelSep = GroupLowResImgIndexSuperpixelSep;
DiffLowResImgIndexSuperpixelSep = GroupDiffLowResImgIndexSuperpixelSep;
maskSky = GroupmaskSky;
maskg = Groupmaskg;
FeatureSuperpixel = GroupFeatureSuperpixel;
save([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat' ],'LowResImgIndexSuperpixelSep');
save([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat' ],'DiffLowResImgIndexSuperpixelSep');
%if Text(j)
%   TextLowResImgIndexSuperpixelSep = GroupTextLowResImgIndexSuperpixelSep;
%   save([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat' ],'TextLowResImgIndexSuperpixelSep');
%end
save([ScratchDataFolder '/data/MaskGSky.mat' ],'maskSky','maskg');
save([ScratchDataFolder '/data/FeatureSuperpixel.mat' ],'FeatureSuperpixel');
end
LowResImgIndexSuperpixelSep = GroupLowResImgIndexSuperpixelSep;
DiffLowResImgIndexSuperpixelSep = GroupDiffLowResImgIndexSuperpixelSep;
maskSky = GroupmaskSky;
maskg = Groupmaskg;
FeatureSuperpixel = GroupFeatureSuperpixel;
save([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat' ],'LowResImgIndexSuperpixelSep');
save([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat' ],'DiffLowResImgIndexSuperpixelSep');
%if Text(j)
%   TextLowResImgIndexSuperpixelSep = GroupTextLowResImgIndexSuperpixelSep;
%   save([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat' ],'TextLowResImgIndexSuperpixelSep');
%end
save([ScratchDataFolder '/data/MaskGSky.mat' ],'maskSky','maskg');
save([ScratchDataFolder '/data/FeatureSuperpixel.mat' ],'FeatureSuperpixel');
