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
function [] = RunCompleteMRF_efficient( Default, img, Predicted, MedSup, Sup, SupOri, TextSup,...
                                        SupNeighborTable, SupSize, maskSky, maskg);
%function [] = RunCompleteMRF(BatchNu,LearnType,LearnSkyEx,LearnLog,LearnNear,...
%                   LearnAlg,baseline,step);

% This Function prepare the Two things for the ReportPlaneParaMRF
% 1) align the superpixel onto the straight line boundary 
%    and the superpixel bounary
% 2) Using Multiple segmentation hurestic to determine occlusion
%    (right now it works sucks)

% Stup Flags
  MultiScaleSup = true; % If yes, analyze Multiple segmentation for relation weights
  STNeeded = true; % If yes, align the ray to the bundary of the superpixels
  DisplayFlag = Default.Flag.DisplayFlag; % set to display or not

% initalize the RayOri and Ray (this is after stitching)
  SampleIndexYSmall = repmat((1:Default.VertYNuDepth)',[1 Default.HoriXNuDepth]);
  SampleIndexXSmall = repmat((1:Default.HoriXNuDepth),[Default.VertYNuDepth 1]);
  SampleImCoordYSmall = (( Default.VertYNuDepth+1-SampleIndexYSmall)-0.5)/Default.VertYNuDepth - Default.Oy_default;
  SampleImCoordXSmall = ( SampleIndexXSmall-0.5)/Default.HoriXNuDepth - Default.Ox_default;
  RayOri = RayImPosition( SampleImCoordYSmall, SampleImCoordXSmall,...
                          Default.a_default, Default.b_default, ...
                          Default.Ox_default,Default.Oy_default); %[ horiXSizeLowREs VertYSizeLowREs 3]
  RayOri = permute(RayOri,[3 1 2]); %[ 3 horiXSizeLowREs VertYSizeLowREs]

%  Img = imread([GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg'],'jpg');
  img = imresize(img, [ Default.SegVertYSize Default.SegHoriXSize]);
  [seglist]=edgeSegDetection(img,i,0); % Run long edge detection
  if DisplayFlag
     DisplaySup(MedSup,300);
     hold on; drawseg(seglist,300);
  end

  % since edge detection using a image size different from the depthmap (55x 305)
  % so do the following coordinate transform
  seglist(:,1:2) = Matrix2ImgCo(Default.SegHoriXSize, Default.SegVertYSize, seglist(:,1:2)); 
  seglist(:,3:4) = Matrix2ImgCo(Default.SegHoriXSize, Default.SegVertYSize, seglist(:,3:4)); 
  seglist(:,1:2) = ImgCo2Matrix(Default.HoriXNuDepth, Default.VertYNuDepth, seglist(:,1:2)); 
  seglist(:,3:4) = ImgCo2Matrix(Default.HoriXNuDepth, Default.VertYNuDepth, seglist(:,3:4)); 

  VB = MedSup(:,round(linspace(1, Default.SegHoriXSize, Default.HoriXNuDepth)));
  HB = MedSup(round(linspace(1, Default.SegVertYSize, Default.VertYNuDepth)),:);

  if STNeeded
	% stitch the regular grid to the seglist
      TextCoor = SupRayAlign( Sup{1}, VB, HB, seglist, SampleIndexXSmall, SampleIndexYSmall, SupSize);
	% transform the TextCoor back into ImgCoord
      SampleImCoordYSmall = (( Default.VertYNuDepth+1-TextCoor(:,:,2))-0.5)/Default.VertYNuDepth - Default.Oy_default;
      SampleImCoordXSmall = ( TextCoor(:,:,1)-0.5)/Default.HoriXNuDepth - Default.Ox_default;
	% generate the New Ray which is stitched into the long lines
      Ray = RayImPosition( SampleImCoordYSmall, SampleImCoordXSmall, ...
                           Default.a_default, Default.b_default, ...
                           Default.Ox_default,Default.Oy_default); %[ horiXSizeLowREs VertYSizeLowREs 3]
      Ray = permute(Ray,[3 1 2]); %[ 3 horiXSizeLowREs VertYSizeLowREs]
  else
      Ray = RayOri;
  end
        
% ===================================END OF STraight Line aligning vertices===========================================

% Multiple segmentation
  if MultiScaleSup ==1
%	MultiScaleSupTable is a table record the group index of sup in higher level of segmentation
     [MultiScaleSupTable] = MultiScalAnalyze( Sup{1}, permute(  cat( 3, Sup{2},...
                            TextSup{1,1}, TextSup{1,2},...
                            TextSup{2,1}, TextSup{2,2},...
                            TextSup{3,1}, TextSup{3,2},...
                            TextSup{4,1}, TextSup{4,2},...
                            TextSup{5,1}, TextSup{5,2},...
                            TextSup{6,1}, TextSup{6,2}),...
                            [3 1 2]));
  else
     MultiScaleSupTable = [];
  end
% =================================END OF Multiple Segmentation Occlusion Decision=====================================
  ReportPlaneParaMRF_Conditioned( Default, Sup{1}, SupOri, full(Predicted.depthMap), zeros( size(Predicted.VarMap)), ...
		RayOri, Ray, SupNeighborTable, maskSky, maskg, MultiScaleSupTable, []); % not really using the Straight line constrains


return;
