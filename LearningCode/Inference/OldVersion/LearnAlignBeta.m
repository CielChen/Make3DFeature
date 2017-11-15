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
function [] =LearnAlignBeta(WeiBatchNumber,LearnOrTest);

FeaflagFull = 1

if nargin < 22
	LearnOrTest = 0;
end
thresh = 90000;


% This funciton Learn Beta for the Alignment term

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
       ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
       Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;
NuPics = size(filename,2);
NuRow = NuRow_default;% 55
batchRow = 1:WeiBatchSize:NuRow;%[1 6 .... 51]
Window = ceil(VertYNuDepth*0.05);

%PicsInd = 1:2:NuPics; % learn only on half of the image
PicsInd = 1:NuPics; % learn only on half of the image
%PicsInd = [10:16 54:62];
%PicsInd = [62];
batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+WeiBatchSize-1,NuRow)
for j = batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+WeiBatchSize-1,NuRow)
%for j = 1:NuRow
  for l = 1:2
   Target = [];
   Feature = [];
    tic;
    for i = PicsInd%1:10%1:NuPics
%        i 
%        load([ScratchDataFolder '/data/SupFea/FeaNList' num2str(i) '.mat']); % load nList (y3 x4 )and FeaNList
        load([ScratchDataFolder '/data/SupFea/FeaNList' num2str(i) 'new.mat']); % load nList (y3 x4 )and FeaNList
        PickMaskY = ceil(nList(:,3)*VertYNuDepth) <= j+Window & ceil(nList(:,3)*VertYNuDepth) > j-Window;
        if l == 1;
           PickMaskOri = abs(nList(:,5:6)*[1 0]') > abs(nList(:,5:6)*[0 1]');% hori
        else
           PickMaskOri = abs(nList(:,5:6)*[1 0]') <= abs(nList(:,5:6)*[0 1]');% vert
        end
        PickMask = PickMaskY & PickMaskOri;
        if FeaflagFull
           disp('full fea')
           Feature = [Feature; FeaNList(PickMask,:)];
%        disp('[1:204 613:end]')
        else
           Feature = [Feature; FeaNList(PickMask,[1:204 613:end])]; %use only abs diff features and neighbors
        end
%        Feature = [Feature; FeaNList(PickMask,1:204)]; %use only abs diff features
%        size(Feature)
        [OccluList]=LaserOccluLabel(i,nList(:,1:2));
%        size(OccluList(PickMask))
        Target = [Target; OccluList(PickMask)];
    end
    TrainOccluPrecent = sum(Target==1)/size(Target,1)
    NuTarget = size(Target,1)

    clear FeaNList nList;
    pack;

%    figure(300);
%    plot(Target);

    if LearnOrTest == 0
	% start learning
	disp('Starting training...........');
        size([Target ones(size(Target))])
        if size([Target ones(size(Target))],1) > thresh
           disp('Target too big');
           pick = randperm(thresh);
           Target = Target(pick,:);
           Feature = Feature(pick,:);
           size([Target ones(size(Target))])
        end
	Psi = glmfit(Feature, [Target ones(size(Target))], 'binomial', 'link', 'logit');
	disp('.......Finished Training');
    else
%	load([ScratchDataFolder '/data/AlignLearn/AlignLearnHori_' num2str(j) '.mat'] );
%	load([ScratchDataFolder '/data/AlignLearnNew/AlignLearnHori_' num2str(j) '.mat'] );
%	load([ScratchDataFolder '/data/AlignLearnAbsDiff/AlignLearnHori_' num2str(j) '.mat'] );
	load([ScratchDataFolder '/data/AlignLearnAbsDiffNei/AlignLearnHori_' num2str(j) '.mat'] );
    end
	
    softPredictedTarget = glmval(Psi, Feature, 'logit');
    predictedTarget = softPredictedTarget > 0.5;

    PositiveAccuracy = sum( (Target == predictedTarget) .* (Target == 1) ) / ...
						sum(  (Target == 1) )
    NegativeAccuracy = sum( (Target == predictedTarget) .* (Target == 0) ) / ...
						sum(  (Target == 0) )
    if l ==1
%       save([ScratchDataFolder '/data/AlignLearn/AlignLearnHori_' num2str(j) '.mat'],...
%            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent');
      if FeaflagFull
       save([ScratchDataFolder '/data/AlignLearnNew/AlignLearnHori_' num2str(j) '.mat'],...
            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
%       save([ScratchDataFolder '/data/AlignLearnAbsDiff/AlignLearnHori_' num2str(j) '.mat'],...
%            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
      else
       save([ScratchDataFolder '/data/AlignLearnAbsDiffNei/AlignLearnHori_' num2str(j) '.mat'],...
            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
      end
    else
%       save([ScratchDataFolder '/data/AlignLearn/AlignLearnVert_' num2str(j) '.mat'],...
%            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent');
      if FeaflagFull
       save([ScratchDataFolder '/data/AlignLearnNew/AlignLearnVert_' num2str(j) '.mat'],...
            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
      else
%       save([ScratchDataFolder '/data/AlignLearnAbsDiff/AlignLearnVert_' num2str(j) '.mat'],...
%            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
       save([ScratchDataFolder '/data/AlignLearnAbsDiffNei/AlignLearnVert_' num2str(j) '.mat'],...
            'Psi','PositiveAccuracy','NegativeAccuracy','TrainOccluPrecent','NuTarget');
      end
    end
    toc
    clear Target Feature Psi;
  end
end
