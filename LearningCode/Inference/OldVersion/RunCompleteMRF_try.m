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
function [] = RunCompleteMRF(BatchNu,LearnType,LearnSkyEx,LearnLog,LearnNear,...
                   LearnAlg,baseline,step);

% selected image with low error as train data set
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

if nargin < 8
   step = [3];
end
%load([ScratchDataFolder '/../MinTest/data/MinTestFileName.mat']);
%load([ScratchDataFolder '/../MinTest/data/MaskGSky.mat']);
load([ScratchDataFolder '/data/MaskGSky.mat']);
previoslyStored = false;
MultiScaleSup = true;
learned = true;
%PicsInd = [10:16 18 54:62];
PicsInd = 1:size(filename,2);
%BatchSize = 5;
BatchSize = 1;
NuPics = size(filename,2);
BatchRow = 1:BatchSize:NuPics;
STNeeded = false
%for i = 1:NuPics
for i = BatchRow(BatchNu):min(BatchRow(BatchNu)+BatchSize-1,NuPics)
%        i
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

        load([ScratchDataFolder '/data/CleanSup/CleanSup' num2str(PicsInd(i)) '.mat']);
        [SegVertYSize, SegHoriXSize] = size(MedSup);
        MedSup = double(MedSup);
        Sup = double(Sup);

        % load
        depthfile = strrep(filename{i},'img','depth_learned'); % the depth filename
        if baseline == 1
           DepthFolder = [ LearnType '_' LearnAlg ...
                          '_Nonsky' num2str(LearnSkyEx) '_Log' num2str(LearnLog) ...
                          '_Near' num2str(LearnNear) '_baseline'];
           load([ScratchDataFolder '/' DepthFolder '/' depthfile '.mat']);
           depthMap = depthMap_base;
        elseif baseline ==2
           DepthFolder = [ LearnType '_' LearnAlg ...
                          '_Nonsky' num2str(LearnSkyEx) '_Log' num2str(LearnLog) ...
                          '_Near' num2str(LearnNear) '_baseline2'];
           load([ScratchDataFolder '/' DepthFolder '/' depthfile '.mat']);
           depthMap = depthMap_base2;
        else
           DepthFolder = [ LearnType '_' LearnAlg ...
                          '_Nonsky' num2str(LearnSkyEx) '_Log' num2str(LearnLog) ...
                          '_Near' num2str(LearnNear)];
           load([ScratchDataFolder '/' DepthFolder '/' depthfile '.mat']);
        end
        LearnedDepth = depthMap; clear depthMap;
  if ~learned 
        depthfile = strrep(filename{i},'img','depth_sph_corr'); % the depth filename
        load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
        LaserDepth = Position3DGrid(:,:,4);
        clear Position3DGrid;         
  end

    % load Var
       Varfile = strrep(filename{i},'img','Var_learned'); % the depth filename
       load([ScratchDataFolder '/Var_' LearnType '_' LearnAlg '_Nonsky' num2str(LearnSkyEx) '_Log' num2str(LearnLog) ...
              '_Near' num2str(LearnNear) '/' Varfile '.mat']);
%        Posi3D = Ray.*repmat(permute(LaserDepth,[3 1 2]),[3 1]);

        % initalize the ray
        RayPorjectImgMapY = repmat((1:SegVertYSize)',[1 SegHoriXSize]);
        RayPorjectImgMapX = repmat((1:SegHoriXSize),[SegVertYSize 1]);
        RayPorjectImgMapY = ((SegVertYSize+1-RayPorjectImgMapY)-0.5)/SegVertYSize - Oy;
        RayPorjectImgMapX = (RayPorjectImgMapX-0.5)/SegHoriXSize - Ox;
        MedRay = RayImPosition(RayPorjectImgMapY,RayPorjectImgMapX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        MedRay = permute(MedRay,[3 1 2]);

        RayPorjectImgMapY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]);
        RayPorjectImgMapX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);
        RayProjImgCo = cat(3, RayPorjectImgMapX, RayPorjectImgMapY);
        RayProjImgCo = permute(RayProjImgCo,[3 1 2]);
        RayProjImgCo = Matrix2ImgCo(HoriXNuDepth, VertYNuDepth, RayProjImgCo(:,:)');
        RayProjImgCo = ImgCo2Matrix(SegHoriXSize, SegVertYSize, RayProjImgCo);
        RayProjImgCo = reshape(RayProjImgCo', 2, VertYNuDepth, []);
        RayPorjectImgMapY = ((VertYNuDepth+1-RayPorjectImgMapY)-0.5)/VertYNuDepth - Oy;
        RayPorjectImgMapX = (RayPorjectImgMapX-0.5)/HoriXNuDepth - Ox;
        RayOri = RayImPosition(RayPorjectImgMapY,RayPorjectImgMapX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        RayOri = permute(RayOri,[3 1 2]);

        % edge detection on the boundary of MedSup
        boundary = conv2(MedSup,[1 -1],'same')~=0 | conv2(MedSup,[1; -1],'same')~=0;
        boundary([1 2 end-1 end],:) = 0;
        boundary(:,[1 2 end-1 end]) = 0;
%        [seglist]=edgeSegDetection(boundary,i,1);
        [GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg']
        Img = imread([GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg'],'jpg');
        Img = imresize(Img, [ SegVertYSize SegHoriXSize]);
        [seglist]=edgeSegDetection(Img,i,0);
        DisplaySup(MedSup,300);
        hold on; drawseg(seglist,300);
%        saveas(300,[ScratchDataFolder '/data/segImg.jpg'],'jpg');
        HBrokeBook = zeros(VertYNuDepth, HoriXNuDepth);
        VBrokeBook = zeros(VertYNuDepth, HoriXNuDepth);
        MovedPatchBook = [];
%        [Sup, MedSup, RayProjImgCo, MovedPatchBook, HBrokeBook, VBrokeBook, StraightLineTable, OccluList] = GenStraightLineFlexibleStick(...
%              seglist,MedSup,Sup, RayProjImgCo, LearnedDepth, [], [], [], Ox, Oy, a , b);
     if STNeeded
        [Sup, MedSup, RayProjImgCo, MovedPatchBook, HBrokeBook, VBrokeBook, StraightLineTable, OccluList] = GenStraightLineFlexibleStickMedSup(...
              seglist,MedSup,Sup, RayProjImgCo, LearnedDepth, [], HBrokeBook, VBrokeBook, Ox, Oy, a , b);
     save([ScratchDataFolder '/data/RayProjImgCo/RayProjImgCo' num2str(i) '.mat'],'RayProjImgCo','seglist','Img','Sup','MedSup','boundary');
        HBrokeBook(:,end) = [];
        VBrokeBook(end,:) = [];
        RayPorjectImgMapY = ((SegVertYSize+1-permute(RayProjImgCo(2,:,:),[2 3 1]))-0.5)/SegVertYSize - Oy;
        RayPorjectImgMapX = (permute(RayProjImgCo(1,:,:),[2 3 1])-0.5)/SegHoriXSize - Ox;
        Ray = RayImPosition(RayPorjectImgMapY,RayPorjectImgMapX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        Ray = permute(Ray,[3 1 2]);
     else
        HBrokeBook(:,end) = [];
        VBrokeBook(end,:) = [];
        OccluList = [];
        StraightLineTable =[];
        Ray = RayOri;
     end

%==================Special Ray Align ment =========================================================================================
 % initalize the RayOri and Ray (this is after stitching)
  SampleIndexYSmall = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]);
  SampleIndexXSmall = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);
  SampleImCoordYSmall = (( VertYNuDepth+1-SampleIndexYSmall)-0.5)/VertYNuDepth - Oy;
  SampleImCoordXSmall = ( SampleIndexXSmall-0.5)/HoriXNuDepth - Ox;
  RayOri = RayImPosition( SampleImCoordYSmall, SampleImCoordXSmall, a, b, Ox, Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
  RayOri = permute(RayOri,[3 1 2]); %[ 3 horiXSizeLowREs VertYSizeLowREs]

%   img = imread([GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg'],'jpg');
%   img = imresize(img, [ SegVertYSize SegHoriXSize]);
%   [seglist]=edgeSegDetection(img,i,0);
%   DisplaySup(MedSup,300);
%   hold on; drawseg(seglist,300);
  seglist(:,1:2) = Matrix2ImgCo(SegHoriXSize, SegVertYSize, seglist(:,1:2));
  seglist(:,3:4) = Matrix2ImgCo(SegHoriXSize, SegVertYSize, seglist(:,3:4));
  seglist(:,1:2) = ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, seglist(:,1:2));
  seglist(:,3:4) = ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, seglist(:,3:4));

  VB = MedSup(:,round(linspace(1, SegHoriXSize, HoriXNuDepth)));
  HB = MedSup(round(linspace(1, SegVertYSize, VertYNuDepth)),:);
  
      TextCoor = SupRayAlign( Sup, VB, HB, seglist, SampleIndexXSmall, SampleIndexYSmall, ones(55,305));
      SampleImCoordYSmall = (( VertYNuDepth+1-TextCoor(:,:,2))-0.5)/VertYNuDepth - Oy;
      SampleImCoordXSmall = ( TextCoor(:,:,1)-0.5)/HoriXNuDepth - Ox;
      Ray = RayImPosition( SampleImCoordYSmall, SampleImCoordXSmall, a, b, Ox, Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
      Ray = permute(Ray,[3 1 2]); %[ 3 horiXSizeLowREs VertYSizeLowREs]

% =================================================================================================================================
%return;
        % Multiple segmentation
     if MultiScaleSup ==1
        load([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat']);
        load([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSepi' num2str(BatchNu) '.mat']);
        DiffSup = DiffLowResImgIndexSuperpixelSep(PicsInd(i),end); clear DiffLowResImgIndexSuperpixelSep;
        TextSup = TextLowResImgIndexSuperpixelSep(PicsInd(i),:,2:end); clear TextLowResImgIndexSuperpixelSep;

        [MultiScaleSupTable] = MultiScalAnalyze( Sup, permute(  cat( 3, DiffSup{1,1},...
                              TextSup{1,1,1}, TextSup{1,1,2},...
                              TextSup{1,2,1}, TextSup{1,2,2},...
                              TextSup{1,3,1}, TextSup{1,3,2},...
                              TextSup{1,4,1}, TextSup{1,4,2},...
                              TextSup{1,5,1}, TextSup{1,5,2},...
                              TextSup{1,6,1}, TextSup{1,6,2}),...
                              [3 1 2]));
     else
        MultiScaleSupTable = [];
     end
 
%        load([ScratchDataFolder '/data/temp/List' num2str(i) '.mat']);
        if learned
%           OccluList = [ 17 18;17 37; 17 97; 17 67; 17 93; 17 266];
%           FitPlaneLearnDepthCoPlane(Sup,MedSup,LearnedDepth, RayOri, Ray, MedRay,...
%           FitPlaneLearnDepthCoPlaneWOPreFit(Sup,MedSup,LearnedDepth, RayOri, Ray, MedRay,...
            Decomp;
%           ReportPlaneParaMRF_Sedumi(step, DepthFolder, Sup,MedSup,LearnedDepth, VarMap, RayOri, Ray, MedRay,...
%               maskSky{PicsInd(i)},maskg{PicsInd(i)},'cvx_allL1Norm',i,...
%               [], OccluList, MultiScaleSupTable, StraightLineTable, HBrokeBook, VBrokeBook,previoslyStored, baseline);
           %ReportPlaneParaMRF(step, DepthFolder, Sup,MedSup,LearnedDepth, VarMap, RayOri, Ray, MedRay,...
           %    maskSky{PicsInd(i)},maskg{PicsInd(i)},'cvx_allL1Norm',i,...
           %    [], OccluList, MultiScaleSupTable, StraightLineTable, HBrokeBook, VBrokeBook,previoslyStored, baseline);
        else
           FitPlaneLaserData_CoPlane(Sup,MedSup,LaserDepth,Ray, MedRay, maskSky{PicsInd(i)},maskg{PicsInd(i)},'cvx_allL1Norm',i,CornerList, CornerList ,previoslyStored);
        end
end
