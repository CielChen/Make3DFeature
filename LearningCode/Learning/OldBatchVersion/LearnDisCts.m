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
function LearnDisCts(LearnAlg,HistFeaType,HistFeaDate,AbsFeaType,AbsFeaDate,WeiBatchNumber,logScale,SkyExclude,LearnNear);
%WeiBatchNumber,logScale,SkyExclude)
% % This function learned the distance

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default WeiBatchSize;

statusFilename = [ClusterExecutionDirectory '/matlabExecutionStatus_ratio.txt'];
% parameters setting
NuPics = size(filename,2);
NuBatch = ceil(NuPics/batchSize);
NuRow = NuRow_default;
% =================== can be modified later==============
SpiatalDisCTSThresh = log(5);
% =======================================================
%Horizon = Horizon_default;
%skyBottom = floor(NuRow/2);
batchRow = 1:WeiBatchSize:NuRow;

% constructing features for each batch of rows from batch featuresa
%load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixel
% load estimated sky
load([ScratchDataFolder '/data/MaskGSky.mat']); % maskg is the estimated ground maskSky is the estimated sky
% load data
load([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat']); % LowResImgIndexSuperpixelSep
load([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat']); % DiffLowResImgIndexSuperpixelSep(medi$large)
%load([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat']); % TextLowResImgIndexSuperpixelSep using Textrea

l = 1;
for i = batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+WeiBatchSize-1,NuRow)
    %i = NuRow
    l
    FeaVectorH = [];
    FeaVectorV = [];
%    FeaWeiH = [];
%    FeaWeiV = [];
    DepthDisCtsVectorH = [];
    DepthDisCtsVectorV = [];
    fid = fopen(statusFilename, 'w+');
    fprintf(fid, 'Currently on row number %i\n', i);
    fclose(fid);	%file opening and closing has to be inside the loop, otherwise the file will not appear over afs
    for j = 1:NuBatch 
        tic
        load([ScratchDataFolder '/data/feature_Abs_' AbsFeaType int2str(j) '_' AbsFeaDate '.mat']); % 'f'
 %       load([ScratchDataFolder '/data/feature_Hist_' HistFeaType int2str(j) '_' HistFeaDate '.mat']); % 'f'
        %toc
        %for k = trainIndex{j}
        for k = 1:size(f,2)%batchSize
            
            %==================
            % load picsinfo just for the horizontal value
            PicsinfoName = strrep(filename{(j-1)*batchSize+k},'img','picsinfo');
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

            % load the depths
            depthfile = strrep(filename{(j-1)*batchSize+k},'img','depth_sph_corr');
            load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
            %==================
            %tic;
            % generate the range of the row for the same thi (weight value)
            RowskyBottom = ceil(NuRow/2);
            PatchSkyBottom = ceil(VertYNuDepth*(1-Horizon));
	    if i <= RowskyBottom
	       PatchRowRatio = PatchSkyBottom/RowskyBottom;
	       RowTop = ceil((i-1)*PatchRowRatio+1);
	       RowBottom = ceil(i*PatchRowRatio);
	    else
	       PatchRowRatio = (VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
  	       RowTop = ceil((i-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
	       RowBottom = ceil((i-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
	    end
            ColumnLeft = 1;
            ColumnRight = HoriXNuDepth;
            
            % generate the DisCts
            [DisCtsH DisCtsV] = DisCtsDetect(Position3DGrid(:,:,4));
            % pick out the 1st end-1 end column and row
            rangeYV = RowTop:RowBottom;
            rangeYH = ColumnLeft:ColumnRight;
            %rangeYV = setdiff(rangeYV,[1 VertYNuDepth-1 VertYNuDepth]);
            rangeXV = RowTop:RowBottom;
            rangeXH = ColumnLeft:ColumnRight;
            %rangeXH = setdiff(rangeXH,[1 HoriXNuDepth-1 HoriXNuDepth]);
            temp = DisCtsV(rangeYV,rangeYH);
            DepthDisCtsVectorV = [DepthDisCtsVectorV; temp(:)];
            temp = DisCtsH(rangeXV,rangeXH);
            DepthDisCtsVectorH = [DepthDisCtsVectorH; temp(:)];

            % pick out the features in the right rows   
            newFeaH = genFeaVector(f{k},[],...
                     rangeXV,rangeXH,(j-1)*batchSize+k,LearnNear,[1]);
            newFeaH = abs(newFeaH - genFeaVector(f{k},[],...
                     rangeXV,(rangeXH+1),(j-1)*batchSize+k,LearnNear,[1]));
            newFeaV = genFeaVector(f{k},[],...
                     rangeYV,rangeYH,(j-1)*batchSize+k,LearnNear,[1]);
            newFeaV = abs(newFeaV - genFeaVector(f{k},[],...
                     (rangeYV+1),rangeYH,(j-1)*batchSize+k,LearnNear,[1]));
% ==================================
        % setting the estimated Ground Verticl Sky segmentation (generated from CMU's code)
        maskEstGVS = 2*ones(VertYNuDepth,HoriXNuDepth);
        maskEstGVS(maskg{i}) = 1;
        maskEstGVS(maskSky{i}) = 3;

            % generate segmentation features
        NuSupType = size(LowResImgIndexSuperpixelSep,2)+size(DiffLowResImgIndexSuperpixelSep,2);
                   %+size(TextLowResImgIndexSuperpixelSep,2)*size(TextLowResImgIndexSuperpixelSep,3);

        for j = 1:NuSupType  % total 21 seg : 3 RGB 6*3= 18 texture filters

            % pick the specific segmentation
            if j==1
                sup = LowResImgIndexSuperpixelSep{i,1};
            elseif j<=3
                sup = DiffLowResImgIndexSuperpixelSep{i,j-1};
            else
                Subm = mod((j-3),6);
                if Subm==0
                   Subm=6;
                end
                sup = TextLowResImgIndexSuperpixelSep{i,Subm,ceil((j-3)/6)};
            end

            % extend the estimated maskGVS to the new segmentation
            NewSupInd = (unique(sup))';
            NewEstGSup = zeros(VertYNuDepth,HoriXNuDepth);
            NewEstVSup = zeros(VertYNuDepth,HoriXNuDepth);
            NewEstSSup = zeros(VertYNuDepth,HoriXNuDepth);
            for m = NewSupInd
                mask = sup == m;
                if any(maskEstGVS(mask)==1) && any(maskEstGVS(mask)==3)
                    GVSInd = analysesupinpatch(maskEstGVS(mask));
                elseif any(maskEstGVS(mask)==1)
                    GVSInd =1;
                elseif any(maskEstGVS(mask)==3)
                    GVSInd =3;
                else
                    GVSInd =2;
                end
                %GVSInd = analysesupinpatch(maskEstGVS(mask));
                if GVSInd == 1
                   NewEstGSup(mask) = m;
                   NewEstVSup(mask) = 0;
                   NewEstSSup(mask) = 0;
                elseif GVSInd == 2
                   NewEstVSup(mask) = m;
                   NewEstGSup(mask) = -1;
                   NewEstSSup(mask) = -1;
                else
                   NewEstSSup(mask) = m;
                   NewEstGSup(mask) = -2;
                   NewEstVSup(mask) = -2;
                end
            end
            %if j == 2
            %   SpreadFactor = gen_SFactor(LearnedDepth,sup,Rz);
            %end
            %clear LowResImgIndexSuperpixelSep;

            % 2nd order smooth
%            [SecXG(j,:) SecYG(j,:)]= gen_2ndSmooth(NewEstGSup);
%            [SecXV(j,:) SecYV(j,:)]= gen_2ndSmooth(NewEstVSup);
%            [SecXS(j,:) SecYS(j,:)]= gen_2ndSmooth(NewEstSSup);

            % 1st order smooth
            [FirstYG(j,:) FirstXG(j,:)] = gen_1stSmooth(NewEstGSup);
            [FirstYV(j,:) FirstXV(j,:)] = gen_1stSmooth(NewEstVSup);
            [FirstYS(j,:) FirstXS(j,:)] = gen_1stSmooth(NewEstSSup);
             FirstY = FirstYG + FirstYV + FirstYS;
             FirstX = FirstXG + FirstXV + FirstXS;
             
                %[GPy{j} ] = gen_GravityP_vertical(maskV);
            %[PlanePriorX PlanePriorY]= gen_PlanePrior(LowResImgIndexSuperpixelSep{i,1});
        end
            
            %size(newFeaH)
            % add on the Hist features
            %SizeHisFea = size(RFVector{k},3);
            %HistFea = genHisFeaVector(reshape(RFVector{k},[],SizeHisFea),...
            %          RowTop,RowBottom,ColumnLeft,ColumnRight,(j-1)*batchSize+k);
            %rank(HistFea)
            %newFeaH = [newFeaH; HistFea];
            %newFeaV = [newFeaV; HistFea];
            
            % finally storage all the data
            [Ix Iy] = meshgrid(rangeXH,rangeXV);
            Index = sub2ind([VertYNuDepth, HoriXNuDepth], Iy(:), Ix(:));
            FeaVectorH =[ FeaVectorH [newFeaH; FirstX(:,Index) ]];
            FeaVectorV =[ FeaVectorV [newFeaV; FirstY(:,Index) ]];
        end
            % get rid of the sky region 
        clear f newFeaH newFeaV DisCtsH DisCtsV Position3DGrid;
        toc
    end

    % learning part
       targetH{l} = DepthDisCtsVectorH(:);
       targetV{l} = DepthDisCtsVectorV(:);
    size(FeaVectorH)
    TsizeH = size(targetH{l},1)
    FsizeH = size(FeaVectorH,1)+1
    TsizeV = size(targetV{l},1)
    FsizeV = size(FeaVectorV,1)+1
    % calculate the weight to even the bias of much more spatial smooth data
    wetH = ones(TsizeH,1);
    wetH(targetH{l}==1) = sum(targetH{l}==0)./sum(targetH{l}==1);
    wetV = ones(TsizeV,1);
    wetH(targetV{l}==1) = sum(targetV{l}==0)./sum(targetV{l}==1);
    
    

    % start learning ========================================
    if strcmp(LearnAlg,'Logit')
       size([targetH{l} ones(TsizeH,1)])
       [thiH{l}] = glmfit(FeaVectorH', [targetH{l} ones(TsizeH,1)],'binomial', 'link', 'logit','weights',wetH);
       X_baseH = ones(TsizeH,1); 
       [thi_baseH{l}] = glmfit(X_baseH, [targetH{l} ones(TsizeH,1)],'binomial', 'link', 'logit','weights',wetH);
       temp =  targetH{l};
       PfitH = glmval(thiH{l},FeaVectorH' , 'logit')>=0.5;
       FNH{l} = sum(PfitH(logical(temp))==0)./sum(logical(temp));
       FPH{l} = sum(PfitH(~logical(temp))==1)./sum(~logical(temp));
       Pfit_baseH = glmval(thi_baseH{l},X_baseH , 'logit') >= 0.5;
       FN_baseH{l} = sum(Pfit_baseH(logical(temp))==0)./sum(logical(temp));
       FP_baseH{l} = sum(Pfit_baseH(~logical(temp))==1)./sum(~logical(temp));
       if TsizeV ~=0
          [thiV{l}] = glmfit(FeaVectorV', [targetV{l} ones(TsizeV,1)],'binomial', 'link', 'logit','weights',wetH);
          X_baseV = ones(TsizeV,1); 
          [thi_baseV{l}] = glmfit(X_baseV, [targetV{l} ones(TsizeV,1)],'binomial', 'link', 'logit','weights',wetH);
          temp =  targetV{l};
          PfitV = glmval(thiV{l},FeaVectorV' , 'logit')>=0.5;
          FNV{l} = sum(PfitV(logical(temp))==0)./sum(logical(temp));
          FPV{l} = sum(PfitV(~logical(temp))==1)./sum(~logical(temp));
          Pfit_baseV = glmval(thi_baseV{l},X_baseV , 'logit') >= 0.5;
          FN_baseV{l} = sum(Pfit_baseV(logical(temp))==0)./sum(logical(temp));
          FP_baseV{l} = sum(Pfit_baseV(~logical(temp))==1)./sum(~logical(temp));
       else
          [thiV{l}] = [thiH{l}];
          [thi_baseV{l}] = [thi_baseH{l}];
          targetV{l} = targetH{l};
          X_baseV = ones(TsizeH,1); 
          PfitV = PfitH;
          FPV{l} = FPH{l};
          FNV{l} = FNH{l};
          Pfit_baseV = Pfit_baseH;
          FP_baseV{l} = FP_baseH{l};
          FN_baseV{l} = FN_baseH{l};
       end

       % calculate error

    elseif strcmp(LearnAlg,'GDA')
    elseif strcmp(LearnAlg,'SVM')
    end
    % end learning ==============================================================================
l = l +1;
DateStamp = date;
save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg ...
      '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
      '_Near' num2str(LearnNear) '_WeiBatNu' num2str(WeiBatchNumber) ...
      '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' ...
      HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '.mat'],...
      'thiH','thiV','thi_baseH','thi_baseH','FPH','FPV','FNH','FNV','FP_baseH','FP_baseV','FN_baseH','FN_baseV');
end

DateStamp = date;
save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg ...
      '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
      '_Near' num2str(LearnNear) '_WeiBatNu' num2str(WeiBatchNumber) ...
      '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' ...
      HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '.mat'],...
      'thiH','thiV','thi_baseH','thi_baseH','FPH','FPV','FNH','FNV','FP_baseH','FP_baseV','FN_baseH','FN_baseV');
%if logScale == 1
%   if SkyExclude == 1
%      save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg '_Nonsky_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '.mat'],'thiH','thiV','thi_baseH','thi_baseH','errorH','errorV','error_baseH','error_baseV','learnRatioH','learnRatioV');
%   else
%      save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg '_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '.mat'],'thiH','thiV','thi_baseH','thi_baseH','errorH','errorV','error_baseH','error_baseV','learnRatioH','learnRatioV');
%   end
%else
%   if SkyExclude == 1
%      save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg '_Nonsky_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '_linear.mat'],'thiH','thiV','thi_baseH','thi_baseH','errorH','errorV','error_baseH','error_baseV','learnRatioH','learnRatioV');
%   else
%      save([ScratchDataFolder '/../learned_parameter/DisCts_' ImgFolder '_' LearnAlg '_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate '_' HistFeaType '_HistFeaDate' HistFeaDate '_LearnDate' DateStamp '_linear.mat'],'thiH','thiV','thi_baseH','thi_baseH','errorH','errorV','error_baseH','error_baseV','learnRatioH','learnRatioV');
%   end
%end
