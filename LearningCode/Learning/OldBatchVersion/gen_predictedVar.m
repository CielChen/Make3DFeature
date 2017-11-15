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
function [] = gen_predictedVar(learningType,SkyExclude,logScale,LearnNear,...
                   LearnAlg,LearnDate,AbsFeaType,AbsFeaDate,HistFeaType,HistFeaDate,...
                   FeaBatchNumber,WeiBatchNumber);
%learningType,logScale,SkyExclude,LearnAlg,AbsFeaType,AbsFeaDate,WeiBatchNumber,logScale,SkyExclude,LearnNear)
% this function generate the learned Var

% =========================================================
%logScale = 1; % for now it's always this case since the variance must be positive
%=========================================================


% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder TrainSet VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

% load estimated sky
load([ScratchDataFolder '/data/MaskGSky.mat']); % maskg is the estimated ground maskSky is the estimated sky

% load superpixel Feadture
load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixel

% load all the  thi in different rows
nut = [];
nut_base = [];
for i = 1:ceil(NuRow_default/WeiBatchSize) % only consider two learning type 'Abs' = Depth 'Fractional' = FractionalRegDepth
    load([ScratchDataFolder '/../learned_parameter/Variance/Var_' TrainSet '_' LearnAlg ...
         '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
         '_Near' num2str(LearnNear) '_WeiBatNu' num2str(i) ...
         '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate' LearnDate '.mat']);
    nut = [nut nu];
    nut_base = [nut_base nu_base];
end  
nut = cell2mat(nut);
nut_base = cell2mat(nut_base);

% mkdir to store the Variance in scratch space
 system(['mkdir ' ScratchDataFolder '/Var_' learningType '_' LearnAlg ...
        '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
         '_Near' num2str(LearnNear)]);
 system(['mkdir ' ScratchDataFolder '/Var_' learningType '_' LearnAlg ...
        '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
         '_Near' num2str(LearnNear) '_baseline']);
%if strcmp(learningType,'Fractional')
%     system(['mkdir ' ScratchDataFolder '/_LearnFDLinearNonSky_']);
%     disp('Fractional')
%else
%  if logScale==1
%     if SkyExclude == 1
%          system(['mkdir ' ScratchDataFolder '/_LearnDLogScaleNonskySep_' learningType]);
%    else
%  	  system(['mkdir ' ScratchDataFolder '/_LearnDLogScale_' learningType]);
%     end
%  else
%     if SkyExclude ==1
%          system(['mkdir ' ScratchDataFolder '/_LearnDNonsky_' learningType]);
%     else
%          system(['mkdir ' ScratchDataFolder '/_LearnD_' learningType]);
%     end
%  end
%end

NuPics = size(filename,2); % number of pictures
NuFeaBatch = ceil(NuPics/batchSize);
for  j =1:NuFeaBatch
%    load([ScratchDataFolder '/data/feature_sqrt_H4_ray' int2str(j) '.mat']); % 'f'
     load([ScratchDataFolder '/data/feature_Abs_Whole' int2str(j) '_.mat']);
    for k = 1:size(f,2)%batchSize
        
        %================
        % load picsinfo just for the horizontal value
        (j-1)*batchSize+k % pics Number

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
        
        % prepare the nuMatrix
        NuRow = NuRow_default;
        for i = 1:NuRow;
            RowskyBottom = ceil(NuRow/2);
            PatchSkyBottom = ceil(VertYNuDepth*(1-Horizon));
            if i <= RowskyBottom
                PatchRowRatio = PatchSkyBottom/RowskyBottom;
                RowTop(i) = ceil((i-1)*PatchRowRatio+1);
                RowBottom(i) = ceil(i*PatchRowRatio);
            else
                PatchRowRatio = (VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
                RowTop(i) = ceil((i-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
                RowBottom(i) = ceil((i-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
            end
        end
        RowNumber = RowBottom'-RowTop'+1;
        nuRow = [];
        nu_baseRow = [];
        for i = 1:NuRow;
            nuRow = [ nuRow nut(:,i*ones(RowNumber(i),1))];
            nu_baseRow = [ nu_baseRow nut_base(:,i*ones(RowNumber(i),1))];
        end    
        %================
        if (j-1)*batchSize+k ==4
            disp('error');
        end    
%        FeaVectorPics = genFeaVector(f{k},FeatureSuperpixel{(j-1)*batchSize+k},...
%                     1,VertYNuDepth,1,HoriXNuDepth,(j-1)*batchSize+k);
        FeaVectorPics = genFeaVector(f{k},FeatureSuperpixel{(j-1)*batchSize+k},...
                     [1:VertYNuDepth],[1:HoriXNuDepth],(j-1)*batchSize+k,0);
        if logScale ==1
            VarMap = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                   repmat(nuRow',[HoriXNuDepth 1]),2),VertYNuDepth,[]));
            VarMap_base = exp(reshape(sum([ones(size(FeaVectorPics,2),1) ].*...
                   repmat(nu_baseRow',[HoriXNuDepth 1]),2),VertYNuDepth,[]));
        else
            size(nuRow)
            size(FeaVectorPics)
            VarMap = reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                   repmat(nuRow',[HoriXNuDepth 1]),2),VertYNuDepth,[]);
            VarMap_base = reshape(sum([ones(size(FeaVectorPics,2),1) ].*...
                   repmat(nu_baseRow',[HoriXNuDepth 1]),2),VertYNuDepth,[]);
        end   
%=====================SkyExclude=====================        
        if SkyExclude ==1
           VarMap(maskSky{(j-1)*batchSize+k}) = max(max(VarMap))+30;
           VarMap_base(maskSky{(j-1)*batchSize+k}) = max(max(VarMap))+30;
        end 
%====================================================
        Varfile = strrep(filename{(j-1)*batchSize+k},'img','Var_learned'); %
        save([ScratchDataFolder '/Var_' learningType '_' LearnAlg ...
        '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
         '_Near' num2str(LearnNear) '/' Varfile '.mat'],'VarMap');
        save([ScratchDataFolder '/Var_' learningType '_' LearnAlg ...
        '_Nonsky' num2str(SkyExclude) '_Log' num2str(logScale) ...
         '_Near' num2str(LearnNear) '_baseline/' Varfile '.mat'],'VarMap_base');
%     if strcmp(learningType,'Fractional')
%        save([ScratchDataFolder '/_LearnFDLinearNonSky_/' Varfile '.mat'], 'VarMap');
%     else 
%        if logScale == 1
%            if SkyExclude == 1
%               save([ScratchDataFolder '/_LearnDLogScaleNonskySep_' learningType '/' Varfile '.mat'],'VarMap');
%            else
%               save([ScratchDataFolder '/_LearnDLogScale_' learningType '/' Varfile '.mat'],'VarMap');
%            end
%        else
%            if SkyExclude == 1
%               save([ScratchDataFolder '/_nLearnDNonsky_' learningType '/' Varfile '.mat'],'VarMap');  
%            else
%               save([ScratchDataFolder '/_LearnD_' learningType '/' Varfile '.mat'],'VarMap');
%            end
%        end 
%     end   
    end             
end    
