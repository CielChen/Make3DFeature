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
function LearnDepth(LearnAlg,AbsFeaType,AbsFeaDate,WeiBatchNumber,logScale,SkyExclude,LearnNear)
% % This function learned the distance

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default WeiBatchSize;

statusFilename = [ClusterExecutionDirectory '/matlabExecutionStatus_depth.txt'];
% parameters setting
NuPics = size(filename,2);
NuBatch = ceil(NuPics/batchSize);
NuRow = NuRow_default;
%Horizon = Horizon_default;
%skyBottom = floor(NuRow/2);
batchRow = 1:WeiBatchSize:NuRow;


l = 1;
for i = batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+WeiBatchSize-1,NuRow)
%for i =  34:35
%i=RowNumber;
    % constructing features for each batch of rows from batch featuresa
    load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixel
    % load estimated sky
    load([ScratchDataFolder '/data/MaskGSky.mat']); % maskg is the estimated ground maskSky is the estimated sky
    l
    FeaVector = [];
    %FeaWei = [];
    DepthVector = [];
    fid = fopen(statusFilename, 'w+');
    fprintf(fid, 'Currently on row number %i\n', i);
    fclose(fid);	%file opening and closing has to be inside the loop, otherwise the file will not appear over afs
    for j = 1:NuBatch 
        tic
        load([ScratchDataFolder '/data/feature_Abs_' AbsFeaType int2str(j) '_' AbsFeaDate '.mat']); % 'f'
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
               
            newFea = genFeaVector(f{k},FeatureSuperpixel{(j-1)*batchSize+k},...
                     [RowTop:RowBottom],[ColumnLeft:ColumnRight],(j-1)*batchSize+k,LearnNear);
            if SkyExclude == 1
                maskSkyPics = maskSky{(j-1)*batchSize+k};
                newFea = newFea(:,~maskSkyPics(RowTop:RowBottom,ColumnLeft:ColumnRight));
            end
            
            if size(newFea,2)~=0
            %    FeaWei = [FeaWei ones(1,size(newFea,2))/size(newFea,2)];
                FeaVector =[ FeaVector newFea];
                depthfile = strrep(filename{(j-1)*batchSize+k},'img','depth_sph_corr');
                load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
                newDepth = genDepthVector(Position3DGrid(:,:,4),...
                    RowTop,RowBottom,ColumnLeft,ColumnRight,(j-1)*batchSize+k);
                if SkyExclude == 1
                   newDepth = newDepth(~maskSkyPics(RowTop:RowBottom,ColumnLeft:ColumnRight),1);
                end
                DepthVector = [DepthVector; newDepth];
            end    
            %DepthVector = [DepthVector genDepthVector(DepthTrueProj{(j-1)*batchSize+k},i,(j-1)*batchSize+k)];
            %l = l + 1;
            %toc;
        end
        clear f newFea Position3DGrid;
        toc
    end

    clear maskg maskSky FeatureSuperpixel maskSkyPics;
    %FeaVector = FeaVector(:,1:round(end*2/3));
    %DepthVector = DepthVector(1:round(end*2/3),:);
    % learning part
    %X{i} = [ones(size(FeaVector,2),1) FeaVector']  % add offset feature to complete the feature set
    if logScale == 1
       target{l} = log(DepthVector);
    else
       target{l} = DepthVector;%log(DepthVector);
    end
    clear DepthVector;
%    whos;
%    pack;
%    whos;
%    pause;
    % full feature learninga
%    [thi{l},stats] = robustfit(FeaVector',target{l},'huber');
    Tsize = size(target{l},1)
    Fsize = size(FeaVector,1)+1
%    A = [-[ones(Tsize,1) FeaVector'] [ones(Tsize,1) FeaVector'] -speye(Tsize) speye(Tsize) sparse(Tsize,Tsize);...
%         -[ones(Tsize,1) FeaVector'] [ones(Tsize,1) FeaVector'] +speye(Tsize) sparse(Tsize,Tsize) -speye(Tsize)];
%    bb = [-target{i};-target{i}];
%    cc = [sparse(Fsize*2,1); ones(Tsize,1); sparse(Tsize*2,1)];
    if strcmp(LearnAlg,'L1norm')
        tic
	cvx_begin
    	cvx_quiet(false);
%    variable thit(Fsize,1);
%    variable k(Tsize,1);
%    variable alpha(Tsize,1);
%    variable beta(Tsize,1);
%    minimize(ones(1,Tsize)*k);
%    A*[thit; k; alpha; beta]==B;
%    alpha>=0;
%    beta<=0;
         [x] = sedumi([-[ones(Tsize,1) FeaVector'] [ones(Tsize,1) FeaVector'] -speye(Tsize) speye(Tsize) sparse(Tsize,Tsize);...
           -[ones(Tsize,1) FeaVector'] [ones(Tsize,1) FeaVector'] +speye(Tsize) sparse(Tsize,Tsize) -speye(Tsize)],...
           [-target{i};-target{i}],[sparse(Fsize*2,1); ones(Tsize,1); sparse(Tsize*2,1)]);
        cvx_end
        thi{l} = x(1:Fsize,1)-x((Fsize+1):2*Fsize,1);
        % base line learning 
        X_base = ones(size(FeaVector,2),1);
	tic
        cvx_begin
           cvx_quiet(true);
           variable thit(1);
           minimize (norm((target{l} - X_base*thit),1));
        cvx_end
        thi_base{l} = thit;
        toc
        % error measure
        error{l} = ( abs( (target{l} - [ones(size(FeaVector,2),1) FeaVector']*thi{l})) );
        error_base{l} = ( abs( (target{l} - X_base*thi_base{l})) );
        learnRatio{l} = sum(error{l})/sum(error_base{l});
    elseif strcmp(LearnAlg,'robustfit')
        tic;
        [thi{l},stats] = robustfit(FeaVector',target{l},'huber');
        toc;
%        pause;
        % base line learning
        X_base = ones(size(FeaVector,2),1);
        [thi_temp,stats] = robustfit(X_base,target{l},'huber');
        thi_base{l} = thi_temp(1);
        % error measure
        error{l} = ( abs( (target{l} - [ones(size(FeaVector,2),1) FeaVector']*thi{l})) );
        error_base{l} = ( abs( (target{l} - X_base*thi_base{l})) );
        learnRatio{l} = sum(error{l})/sum(error_base{l});
    elseif strcmp(LearnAlg,'L2norm')
        thi{l} = [ones(Tsize,1) FeaVector']\target{l}; 
        % base line learning
        X_base = ones(size(FeaVector,2),1);
        thi_base{l} = X_base\target{l};
        % error measure
        error{l} = ( ( (target{l} - [ones(size(FeaVector,2),1) FeaVector']*thi{l})).^2 );
        error_base{l} = ( ( (target{l} - X_base*thi_base{l})).^2 );
        learnRatio{l} = sqrt(sum(error{l})/sum(error_base{l}));
    end
    
    
l = l +1;
DateStamp = date;
save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg ...
      '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
      '_Near' num2str(LearnNear) '_WeiBatNu' num2str(WeiBatchNumber) ...
      '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '.mat'],...
      'thi','thi_base','error','error_base','learnRatio');
end

DateStamp = date;
save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg ...
      '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
      '_Near' num2str(LearnNear) '_WeiBatNu' num2str(WeiBatchNumber) ...
      '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '.mat'],...
      'thi','thi_base','error','error_base','learnRatio');
%if logScale == 1
%   if SkyExclude == 1
%      save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg '_Nonsky_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '.mat'],'thi','thi_base','error','error_base','learnRatio');
%   else
%      save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg '_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '.mat'],'thi','thi_base','error','error_base','learnRatio');
%   end
%else
%   if SkyExclude == 1
%      save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg '_Nonsky_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '_linear.mat'],'thi','thi_base','error','error_base','learnRatio');
%   else
%      save([ScratchDataFolder '/../learned_parameter/Depth_' ImgFolder '_' LearnAlg '_WeiBatNu' num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' DateStamp '_linear.mat'],'thi','thi_base','error','error_base','learnRatio');
%   end
%end
