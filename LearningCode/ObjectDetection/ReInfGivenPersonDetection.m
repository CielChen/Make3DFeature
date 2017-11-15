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
function [] = ReInfGivenPersonDetection( defaultPara, ImgName)

% This function Re-Inference after given Person Detection info
TimeStart = tic;

% initialized parameters
OutPutFolder = [[defaultPara.Fdir '/Wrl/' ImgName '/']];
system(['mkdir ' OutPutFolder]);
taskName = 'ObjectDetectionRemoveDepth';%(Not Used) taskname will append to the imagename and form the outputname
Flag.DisplayFlag = 0;
Flag.IntermediateStorage = 0;
Flag.FeaturesOnly = 0;
Flag.FeatureStorage = 0; %Min add May 2nd
Flag.NormalizeFlag = 1;
Flag.BeforeInferenceStorage = 1;
Flag.NonInference = 0;
Flag.AfterInferenceStorage = 1;
ScratchFolder = [defaultPara.Fdir '/data/' ImgName]%IMG_0614'; % ScratchFolder
system(['mkdir ' ScratchFolder]);
ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/'
Default = SetupDefault_New(...
	[ ImgName '_' taskName],...
	ParaFolder,...
	OutPutFolder,...
	ScratchFolder,...
	Flag);

% extract the Person Detection Box ========================================================
disp(' Extract the Person Detection Box');
ImgPersonBox = imread([ defaultPara.Fdir '/Person/' ImgName '.jpg']);
ImgPersonBoxSize = size( ImgPersonBox);
ImgPersonBoxSize = ImgPersonBoxSize(1:2);
Img = imread([ defaultPara.Fdir '/jpg/' ImgName '.jpg']);
ImgResize = imresize(Img, ImgPersonBoxSize, 'nearest');
fid = fopen([defaultPara.Fdir '/Person/' ImgName '.txt']);
%fgetl(fid);% skip first line which is all zeros
PersonBox = fscanf(fid, '%f', [defaultPara.NumCol, inf]);
PersonBox = PersonBox(1:4,:);
PersonBox = PersonBox';
% debug purpose only------------------------------
% PersonBox(:,4) = PersonBox(:,4) - 150;
% ------------------------------------------------
PersonBox = [ PersonBox(:, [ 2 1]) PersonBox(:, [2 1])+PersonBox(:,[4 3])]; % Data Format [Y_Top X_Right Y_Bottom X_Left]
OutLier = PersonBox(:,3) < ImgPersonBoxSize(1)/4; % remove Squares too high
PersonBox( OutLier, :) = [];
[C I] = sort(PersonBox(:,3));% sort the Box from far to close
PersonBox = PersonBox(I,:);
% PersonBox(1,:) = [];
PersonBox
%return;
NumBox = size(PersonBox,1);
if defaultPara.Flag.Disp
	PlotBox( ImgResize, PersonBox);
end
disp([ '		' num2str(toc( TimeStart))]);
% pause
% =========================================================================================


% load Mono-Model info ====================================================================
load([defaultPara.Fdir '/data/' ImgName '/' ImgName '__AInfnew.mat']);
SupSize = size(Sup);

SupO = Sup;
% return;
% =========================================================================================

MAGIC_NUMBER_SUP_REJECT_RATIO = 0.5;
% Refine Person Contour and Modify Superpixel ==========================================
disp('Refine Person Contour and Modify Superpixel');
PersonBoxINSupSize = max( round( PersonBox./repmat( ImgPersonBoxSize./SupSize , NumBox, 2) ), 1);
PersonBoxINSupSize(:,[ 1 3]) = min( PersonBoxINSupSize(:,[ 1 3]), SupSize(1));
PersonBoxINSupSize(:,[ 2 4]) = min( PersonBoxINSupSize(:,[ 2 4]), SupSize(2));
PersonBoxINSupSize = RefineBox(Sup, PersonBoxINSupSize, MAGIC_NUMBER_SUP_REJECT_RATIO);

% prevent the Box in the last row =======
DetectMark = PersonBoxINSupSize(:,3) >= (size(Sup,1) -1);
PersonBoxINSupSize(DetectMark,3) = (size(Sup,1) -1);
% ===================================
% =======================================

PersonBox = round(PersonBoxINSupSize./repmat( size(Sup)./ImgPersonBoxSize,NumBox,2));
if defaultPara.Flag.Disp
	PlotBox( ImgResize, PersonBox,12);
end
%SE = strel('octagon',3);  
NHOOD = [[0 1 0];[1 1 1];[0 0 0]];
%HEIGHT = NHOOD;
%SE = strel('arbitrary',NHOOD,HEIGHT);  
for i = 1:NumBox
	PersonMask = logical( zeros(size(Sup)) );
	PersonMask( PersonBoxINSupSize(i,1): PersonBoxINSupSize(i,3), ...
		    PersonBoxINSupSize(i,2): PersonBoxINSupSize(i,4)) = true;
	Sup( PersonMask) = -i;
	SupO( PersonMask) = -i;
	% boundary set to zero
    PersonMask_boundry = imdilate( PersonMask, NHOOD);
	PersonMask_boundry(PersonMask) = 0; 
	PersonMask_boundry = logical(PersonMask_boundry);
	figure(11); imagesc(PersonMask_boundry);
	%pause
	Sup(PersonMask_boundry) = 0;
	SupO(PersonMask_boundry) = -i;
end
% SupOri = Sup;
% return;
disp(['		' num2str(toc( TimeStart))]);
% =========================================================================================


% Using Person contour info ===============================================================
% 1) Modified MultiScaleSupTable SupNeighborTable	
MinSupInd = min(Sup(:));% might be negative
MultiScaleSupTable  = [ zeros(MinSupInd, 14);MultiScaleSupTable];
disp('Modified SupNeighborTable');
SupNeighborTable = FastSupNeighborTable(Sup);
disp(['		' num2str(toc( TimeStart))]);
% =========================================================================================

% Re-Inference ============================================================================
% need Default Predicted.VarMap
ReportPlaneParaMRF_Conditioned_ObjectDetection( Default, 3, [],... 
	Sup, SupO, MedSup, full(depthMap), zeros( size( depthMap)), RayOri, Ray, ... % big change
	SupNeighborTable, [], maskSky, maskG,...
	[], [], MultiScaleSupTable, [], [], [], false, 0);
% =========================================================================================

% Constructing Model ======================================================================

% =========================================================================================
%return;
