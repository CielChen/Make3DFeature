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
function []=PredictDepthMRF(DepthDirectory,logScale,SkyExclude,Lazer,BatchNu,step)
%this function generate the predicted plane


DepthDirectory
%return;
% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

% initialize parameter
NuPatch = VertYNuDepth*HoriXNuDepth;

% load data
load([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat']); % load LowResImgIndexSuperpixelSep
load([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat']); % load DiffLowResImgIndexSuperpixelSep(medi$large)
load([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat']); % load TextLowResImgIndexSuperpixelSep using Textrea
load([ScratchDataFolder '/data/MaskGSky.mat']); % load maskg maskSky from CMU's output

% prepare to store the predictedM
%[ScratchDataFolder '/_predicted_' DepthDirectory]
%system(['mkdir ' ScratchDataFolder '/_predicted_' DepthDirectory]);
% set parameter
ZTiltFactor = 1; % both can be estimated after group fit that estimate the Norm_floor
YTiltFactor = 1;

% initial parameter
BatchSize = 5;
NuPics = size(filename,2);
BatchRow = 1:BatchSize:NuPics;
%for i = 10% : NuPics
BatchRow(BatchNu):min(BatchRow(BatchNu)+BatchSize-1,NuPics)
for i = BatchRow(BatchNu):min(BatchRow(BatchNu)+BatchSize-1,NuPics)
        i 
%return;
        % load picsinfo just for the horizontal value
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
 
        % load depthMap
        depthfile = strrep(filename{i},'img','depth_learned'); % the depth filename
        load([ScratchDataFolder '/' DepthDirectory '/' depthfile '.mat']);
%        if logScale == 1
%            if SkyExclude ==1
%                load([ScratchDataFolder '/_LearnDLogScaleNonsky_' DepthDirectory '/' depthfile '.mat']);
%            else
%                load([ScratchDataFolder '/_LearnDLogScale_' DepthDirectory '/' depthfile '.mat']);
%            end
%        else
%            if SkyExclude ==1
%                load([ScratchDataFolder '/_LearnD_' DepthDirectory '/' depthfile '.mat']);
%            else
%                load([ScratchDataFolder '/_LearnD_' DepthDirectory '/' depthfile '.mat']);
%            end
%        end
%        depthMap
%return;            
        LearnedDepth = depthMap; clear depthMap;

        % load MediResImgIndexSuperpixelSep
        load([ScratchDataFolder '/data/MedSeg/MediResImgIndexSuperpixelSep' num2str(i) '.mat']);
        MedSup = MediResImgIndexSuperpixelSep;

        % load Lazer depthMap
        if Lazer == 1
           depthfile = strrep(filename{i},'img','depth_sph_corr'); % the depth filename
           load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
           LaserDepth = Position3DGrid(:,:,4);
           clear Position3DGrid; 
        end      

if false
        % generate straight line for straight line prior
        Img = imread([GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg'],'jpg');
        if  prod(size(Img))> SegVertYSize*SegHoriXSize*3
            Img = imresize(Img,[SegVertYSize SegHoriXSize],'nearest');
        end
        [ImgYSize ImgXSize dummy] = size(Img);
        [lineSegList] = edgeSegDetection(Img,i,0);

% =================================================================================

        NuSeg = size(lineSegList,1);
%================================This transform might cause the offset===================================
        % Normalize the lineSegList to the size of VertYNuDepth HoriXNuDepth (watch out for the +- 0.5 offset)
        lineSegList = ((lineSegList+0.5)./repmat([ImgXSize ImgYSize ImgXSize ImgYSize],NuSeg,1)...
                      .*repmat([HoriXNuDepth VertYNuDepth HoriXNuDepth VertYNuDepth],NuSeg,1))-0.5;
%========================================================================================================

        % take out the vertical edge so that we may combine it with 1stSmooth in Y direction
        % < 0.9 like to deal with VertLine cause VerLine don't stick 
        temp = (abs(round(lineSegList(:,1))-round(lineSegList(:,3)))./abs(round(lineSegList(:,2))-round(lineSegList(:,4))));
        VertLineSegList = lineSegList( temp < 0.9  ,:);% Verti Straight Line
        lineSegList = lineSegList( temp >= 0.9  ,:);% NonVerti Straight Line 
        
        % Start processing the Line
        MovedPatchBook = []; % keep the record of Patch been moved (a patch can ouly be moved once)
        RayPorjectImgMapY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]); % keep the new Ray in Y position
        RayPorjectImgMapX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);
        VBrokeBook = sparse(VertYNuDepth,HoriXNuDepth);
        HBrokeBook = sparse(VertYNuDepth,HoriXNuDepth);
        RayPorjectImgMapY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]); % keep the new Ray in Y position
        RayPorjectImgMapX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);        %First process the nonVerti
        [StraightLinePriorSelectMatrix RayPorjectImgMapY RayPorjectImgMapX MovedPatchBook LineIndex HBrokeBook VBrokeBook] = ...
         StraightLinePrior(i,Img,lineSegList,LearnedDepth,RayPorjectImgMapY,RayPorjectImgMapX,MovedPatchBook,HBrokeBook,VBrokeBook,0,a,b,Ox,Oy);
        %process the Verti
        % first find out VertWall and VertiGround
        disp('Verti') 
        [StraightLinePriorSelectMatrixVert RayPorjectImgMapY RayPorjectImgMapX MovedPatchBook LineIndexVert HBrokeBook ... 
         VBrokeBook] = StraightLinePrior(i,Img,VertLineSegList,LearnedDepth,RayPorjectImgMapY,RayPorjectImgMapX,...
         MovedPatchBook,HBrokeBook,VBrokeBook,1,a,b,Ox,Oy);
end
%        VertLineSegList = round(VertLineSegList);
%        NuVertSeg = size(VertLineSegList,1);
%        VertLineMask = zeros(VertYNuDepth, HoriXNuDepth);
%        for k = 1:NuVertSeg
%            if VertLineSegList(k,2) < VertLineSegList(k,4)
%               VertLineMask(max(VertLineSegList(k,2),1):min((VertLineSegList(k,4)-1),VertYNuDepth),...
%                            min(max(VertLineSegList(k,1),1),HoriXNuDepth)) = 1;
%            else
%               VertLineMask(max(VertLineSegList(k,4),1):min((VertLineSegList(k,2)-1),VertYNuDepth),...
%                            min(max(VertLineSegList(k,1),1),HoriXNuDepth)) = 1;
%            end
%        end
         
%         
        % generate ray porjection position in the image plane
        RayPorjectImgMapY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]); % keep the new Ray in Y position
        RayPorjectImgMapX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);        %First process the nonVerti
        RayPorjectImgMapY = ((VertYNuDepth+1-RayPorjectImgMapY)-0.5)/VertYNuDepth - Oy; 
        RayPorjectImgMapX = (RayPorjectImgMapX-0.5)/HoriXNuDepth - Ox;
        % generate specific ray for whole pics
        RayCenter = RayImPosition(RayPorjectImgMapY,RayPorjectImgMapX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        Rx = RayCenter(:,:,1);
        Rx = Rx(:);
        Ry = RayCenter(:,:,2);
        Ry = Ry(:);
        Rz = RayCenter(:,:,3);
        Rz = Rz(:); 

%=====================================================        
        % MRF reshape the 3d cloud

%============================
        % setting the estimated Ground Verticl Sky segmentation (generated from CMU's code)
        maskEstGVS = 2*ones(VertYNuDepth,HoriXNuDepth);
        maskEstGVS(maskg{i}) = 1;
        maskEstGVS(maskSky{i}) = 3;
        %GSize = sum(maskg{i});
        %SkySize = sum(maskSky{i});
%============================

        NuSupType = size(LowResImgIndexSuperpixelSep,2)+size(DiffLowResImgIndexSuperpixelSep,2)...
                   +size(TextLowResImgIndexSuperpixelSep,2)*size(TextLowResImgIndexSuperpixelSep,3);

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

            % Exclude the sky first
            sup(maskEstGVS == 3) = 0; 
            % extend the estimated maskGVS to the new segmentation
            NewSupInd = setdiff((unique(sup))',0);
            NewEstGSup = zeros(VertYNuDepth,HoriXNuDepth);
            NewEstVSup = zeros(VertYNuDepth,HoriXNuDepth);
            for m = NewSupInd
                mask = sup == m;
                if any(maskEstGVS(mask)==1)
                    GVSInd =1;
                else
                    GVSInd =2;
                end
                %GVSInd = analysesupinpatch(maskEstGVS(mask));
                if GVSInd == 1
                   NewEstGSup(mask) = m;
                   NewEstVSup(mask) = 0;
                elseif GVSInd == 2
                   NewEstVSup(mask) = m;  
                   NewEstGSup(mask) = -1;  
                end
            end 
            %if j == 2
            %   SpreadFactor = gen_SFactor(LearnedDepth,sup,Rz);
            %end
            %clear LowResImgIndexSuperpixelSep;
            
            % 2nd order smooth
            [SecXG(j,:) SecYG(j,:)]= gen_2ndSmooth(NewEstGSup);
            [SecXV(j,:) SecYV(j,:)]= gen_2ndSmooth(NewEstVSup);

            % 1st order smooth
            [FirstYG(j,:) FirstXG(j,:)] = gen_1stSmooth(NewEstGSup);
            [FirstYV(j,:) FirstXV(j,:)] = gen_1stSmooth(NewEstVSup);
            	%[GPy{j} ] = gen_GravityP_vertical(maskV);
            %[PlanePriorX PlanePriorY]= gen_PlanePrior(LowResImgIndexSuperpixelSep{i,1});
        end 
 
        % set weight for different segmentation
        small=10; med=5; large=1;
%        small=5; med=2.5; large=0.5;
        temp =[small; med; large; small*ones(6,1); med*ones(6,1) ;large*ones(6,1)];
        Wei2ndSmoothGX = temp;
        Wei2ndSmoothGY = temp;
        Wei2ndSmoothVX = temp;
        Wei2ndSmoothVY = temp;
%        Wei1stSmoothGX = temp*10;
%        Wei1stSmoothGY = temp*10; % group can have slope
        Wei1stSmoothVX = temp; % vertical wall can have orientation
        Wei1stSmoothGX = temp;
        Wei1stSmoothGY = temp; % group can have slope
        Wei1stSmoothVY = temp;
        
        % set weight for the VertLineSeg Ground and VertialWalls
%        WeiLinSegVert = 11;

        % set weight for the straight line prior
%        WeiStraightLine = 1120;

        % generate the smooth matrix
        M2ndSmoothY = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],[-1 0 1],NuPatch,NuPatch);
        M2ndSmoothX = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],...
                              [-VertYNuDepth 0 VertYNuDepth],NuPatch,NuPatch);
        M1stSmoothY = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 1],NuPatch,NuPatch);
        M1stSmoothX = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 VertYNuDepth],NuPatch,NuPatch);

        % generate beta
%        beta2ndSmoothGX = Wei2ndSmoothGX'*ones(21,1) * logisticResponse(Wei2ndSmoothGX'*SecXG / (Wei2ndSmoothGX'*ones(21,1)) );
        beta2ndSmoothGX = Wei2ndSmoothGX'*SecXG;
%        beta2ndSmoothGX([cell2mat(LineIndex(:)') cell2mat(LineIndexVert(:)')])=0;
        beta2ndSmoothGY = Wei2ndSmoothGY'*SecYG;
%        beta2ndSmoothGY([cell2mat(LineIndex(:)') cell2mat(LineIndexVert(:)')])=0;
%        beta2ndSmoothGY = Wei2ndSmoothGY'*ones(21,1) * logisticResponse(Wei2ndSmoothGY'*SecYG / (Wei2ndSmoothGY'*ones(21,1)) );
%        beta2ndSmoothVX = Wei2ndSmoothVX'*ones(21,1) * logisticResponse(Wei2ndSmoothVX'*SecXV / (Wei2ndSmoothVX'*ones(21,1)) );
        beta2ndSmoothVX = Wei2ndSmoothVX'*SecXV;
%        beta2ndSmoothVX([cell2mat(LineIndex(:)') cell2mat(LineIndexVert(:)')])=0;
%        beta2ndSmoothVY = Wei2ndSmoothVY'*ones(21,1) * logisticResponse(Wei2ndSmoothVY'*SecYV / (Wei2ndSmoothVY'*ones(21,1)) );
        beta2ndSmoothVY = Wei2ndSmoothVY'*SecYV;
%        beta2ndSmoothVY([cell2mat(LineIndex(:)') cell2mat(LineIndexVert(:)')])=0;
%        beta1stSmoothGY = Wei1stSmoothGY'*ones(21,1)*logisticResponse(Wei1stSmoothGY'*FirstYG/(Wei1stSmoothGY'*ones(21,1)) );
        beta1stSmoothGY = Wei1stSmoothGY'*FirstYG;
%        beta1stSmoothGY([cell2mat(LineIndex(:,1)') cell2mat(LineIndexVert(:,1)')])=0;
%        beta1stSmoothVY = Wei1stSmoothVY'*ones(21,1)*logisticResponse(Wei1stSmoothVY'*FirstYV/(Wei1stSmoothVY'*ones(21,1)) );
        beta1stSmoothVY = Wei1stSmoothVY'*FirstYV;
%        beta1stSmoothVY([cell2mat(LineIndex(:,1)') cell2mat(LineIndexVert(:,1)')])=0;
%        beta1stSmoothSY = Wei1stSmoothSY'*ones(21,1)*logisticResponse(Wei1stSmoothSY'*FirstYS/(Wei1stSmoothSY'*ones(21,1)) );
%        beta1stSmoothGX = Wei1stSmoothGX'*ones(21,1)*logisticResponse(Wei1stSmoothGX'*FirstXG/(Wei1stSmoothGX'*ones(21,1)) );
        beta1stSmoothGX = Wei1stSmoothGX'*FirstXG;
%        beta1stSmoothGX([cell2mat(LineIndex(:,3)') cell2mat(LineIndexVert(:,3)')])=0;
%        beta1stSmoothVX = Wei1stSmoothVX'*ones(21,1)*logisticResponse(Wei1stSmoothVX'*FirstXV/(Wei1stSmoothVX'*ones(21,1)) );
        beta1stSmoothVX = Wei1stSmoothVX'*FirstXV;
%        beta1stSmoothVX([cell2mat(LineIndex(:,3)') cell2mat(LineIndexVert(:,3)')])=0;
%        beta1stSmoothSX = Wei1stSmoothSX'*ones(21,1)*logisticResponse(Wei1stSmoothSX'*FirstXS/(Wei1stSmoothSX'*ones(21,1)) );
        
        % generate Q
        tempMask = beta2ndSmoothGX == 0;
        NuElement = sum(~tempMask);
        Q2ndXy = spdiags((beta2ndSmoothGX(~tempMask))'...
                 ,0,NuElement,NuElement)*M2ndSmoothX(~tempMask,:)*spdiags(Ry,0,NuPatch,NuPatch);

        tempMask = beta2ndSmoothVX == 0;
        NuElement = sum(~tempMask);
        Q2ndXz = spdiags((beta2ndSmoothVX(~tempMask))'...
                 ,0,NuElement,NuElement)*M2ndSmoothX(~tempMask,:)*spdiags(Rz,0,NuPatch,NuPatch);  

        tempMask = beta2ndSmoothGY == 0;
        NuElement = sum(~tempMask);
        Q2ndYy = spdiags((beta2ndSmoothGY(~tempMask))'...
                 ,0,NuElement,NuElement)*M2ndSmoothY(~tempMask,:)*spdiags(Ry,0,NuPatch,NuPatch); 

        tempMask = beta2ndSmoothVY == 0;
        NuElement = sum(~tempMask);
        Q2ndYz = spdiags((beta2ndSmoothVY(~tempMask))'...
                 ,0,NuElement,NuElement)*M2ndSmoothY(~tempMask,:)*spdiags(Rz,0,NuPatch,NuPatch); 

%        Q2ndLine = WeiStraightLine*StraightLinePriorSelectMatrix;

%        Q2ndVertLine = WeiLinSegVert*StraightLinePriorSelectMatrixVert;
         
        tempMask = beta1stSmoothVY == 0;
        NuElement = sum(~tempMask);
        Q1stYz = spdiags((beta1stSmoothVY(~tempMask))'...
                 ,0,NuElement,NuElement)*M1stSmoothY(~tempMask,:)*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);

        tempMask = beta1stSmoothVX == 0;
        NuElement = sum(~tempMask);
        Q1stXz = spdiags((beta1stSmoothVX(~tempMask))'...
                 ,0,NuElement,NuElement)*M1stSmoothX(~tempMask,:)*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);

        tempMask = beta1stSmoothGY == 0;
        NuElement = sum(~tempMask);
        Q1stYy = spdiags((beta1stSmoothGY(~tempMask))'...
                 ,0,NuElement,NuElement)*M1stSmoothY(~tempMask,:)*spdiags(Ry*YTiltFactor,0,NuPatch,NuPatch);

        tempMask = beta1stSmoothGX == 0;
        NuElement = sum(~tempMask);
        Q1stXy = spdiags((beta1stSmoothGX(~tempMask))'...
                 ,0,NuElement,NuElement)*M1stSmoothX(~tempMask,:)*spdiags(Ry*YTiltFactor,0,NuPatch,NuPatch);
% ============================START MRF OPTMIZATIOM=========================================================
         %=================
% lineprog ===========================================
%         B = [DMatrixY;spdiags(maskO(:),[0],NuPatch,NuPatch)];
%         Q = [Q2ndXz;Q2ndYz];
%         A = [B -speye(size(B,1)) sparse(size(B,1),size(Q,1));...
%             -B -speye(size(B,1)) sparse(size(B,1),size(Q,1));...
%              Q sparse(size(Q,1),size(B,1)) -speye(size(Q,1));...
%             -Q sparse(size(Q,1),size(B,1)) -speye(size(Q,1));...
%             -speye(NuPatch) sparse(NuPatch,size(B,1)+size(Q,1))];
%         bb = [B*LearnedDepth(:); - B*LearnedDepth(:);...
%               sparse(size(Q,1)*2,1); -5*ones(NuPatch,1)];
%         f = [sparse(NuPatch,1); ones(size(B,1)+size(Q,1),1)];
%         x = linprog(f,A,bb);
%=====================================================

% quadprog ===========================================
% only 2nd order smooth use L1 norm
%        Q1size = size([Q1stXy;Q1stYy],1)% Q1stYz;Q1stXz],1)
        Q1size = size([Q1stXy;Q1stYy; Q1stYz;Q1stXz],1)
%        Q1size = size([Q1stXy;Q1stYy; Q1stYz;Q1stXz],1)
%        SLsize = size([Q2ndLine;Q2ndVertLine],1)
%        Q2size = size([Q2ndXz;Q2ndYz],1)
        Q2size = size([Q2ndXz;Q2ndYy;Q2ndYz;Q2ndXy],1)
        size([Q1stXy;Q1stYy;Q1stYz;Q1stXz])% Q1stYz;Q1stXz],1)
%        size([Q2ndLine])
        size([Q2ndXz;Q2ndYy;Q2ndYz;Q2ndXy])
%	E = [Q1stYy ;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);...
%              Q2ndLinez];
%        e = [sparse(Q1size,1); -LearnedDepth(:); sparse(SLsize,1)];
%        B = [Q2ndXz ; Q2ndYz];
%	H = 2*[[Q1stYy;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);Q2ndLinez]'*...
%             [Q1stYy;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);Q2ndLinez] sparse(NuPatch,Q2size*3);...
%              sparse(Q2size*3,NuPatch+Q2size*3)];
%        f =[2*[sparse(Q1size,1); -LearnedDepth(:); sparse(SLsize,1)]'*...
%            [Q1stYy;Q1stYz; Q1stXz...
%             spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch); Q2ndLinez] ones(1,Q2size) sparse(1,Q2size*2)]';
%        Aeq = [[Q2ndXz ; Q2ndYz] -spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
%               spdiags(ones(Q2size,1),[0],Q2size,Q2size) sparse(Q2size,Q2size);...
%              [Q2ndXz ; Q2ndYz] spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
%               sparse(Q2size,Q2size) spdiags(ones(Q2size,1),[0],Q2size,Q2size)];
%        beq = sparse(Q2size*2,1);
%        lb = [5*ones(NuPatch,1);sparse(Q2size*2,1);...
%              -2*max(LearnedDepth(:))*ones(Q2size,1)];
%        ub = [max(LearnedDepth(:))*ones(NuPatch,1);...
%              2*max(LearnedDepth(:))*ones(Q2size*2,1); sparse(Q2size,1)];
%        options = optimset('LargeScale', 'on', 'Display', 'off');
%        x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
%        predictedM = [spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)...
%                      sparse(NuPatch,Q2size*3)]*x;
%        A = [sparse(Q2size*2,NuPatch+Q2size)...
%             spdiags([-ones(Q2size,1);ones(Q2size,1)],[0],Q2size*2,Q2size*2)];
%        b =sparse(Q2size*2,1);
%=====================================================
%         predictedM = spdiags([ones(1,NuPatch) sparse(1,size(B,1)+size(Q,1))]',0,NuPatch...
%                             ,NuPatch+size(B,1)+size(Q,1))*x; 
%        predictedM = (LearnedDepth(:));


  	     %[[Q2ndYz] -spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
             %   spdiags(ones(Q2size,1),[0],Q2size,Q2size) sparse(Q2size,Q2size);...
             %  [Q2ndYz] spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
             %   sparse(Q2size,Q2size) spdiags(ones(Q2size,1),[0],Q2size,Q2size)]*[predictedM; kt; ga; al] == 0;
              
%             [Q2ndXz; Q2ndYz]*predictedM <=kt;
%             [Q2ndXz; Q2ndYz]*predictedM >= -kt;
%             predictedM>=5;
              %ga>=0;
              %al<=0;
%         cvx_end   
%         toc;

% post prosessing on straight line
% second optimization
% predictedM = [Q2ndLinez;spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)]\[sparse(SLsize,1); -LearnedDepth(:)];
% predictedM = full(predictedM);
        Date =date;
%        predictedM =reshape(predictedM,VertYNuDepth,[]);
         % quadprog Start
  % Yalmip Start ===============================
  mask = LearnedDepth(:)<80;
  AssociateM = zeros(NuPatch,1);
  AssociateM(mask) = 1;
  %ClearLearnedDepth = LearnedDepth(:);
  %ClearLearnedDepth(~mask) = 0;
  Name{1} = '1stOSmooth';
  Name{2} = '2ndOSmooth';
  for m =step
%  for m =1
if true
    m
    opt = sdpsettings('solver','sedumi');
    predictedM = sdpvar(NuPatch,1);
    F = set(predictedM>=0)+set(predictedM<=80);
    if m == 1
       solvesdp(F,norm( [Q1stXy;Q1stYy; Q1stYz;Q1stXz;...
                    spdiags(AssociateM,[0],NuPatch,NuPatch)]...
                    *predictedM+[sparse(Q1size,1);-LearnedDepth(:)],1)...
               , opt);
%                    spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)]...
    else
  if false
       solvesdp(F,norm( [Q2ndXz;Q2ndYy;Q2ndYz;Q2ndXy;...
                    spdiags(AssociateM,[0],NuPatch,NuPatch)]...
                    *predictedM+[sparse(Q2size,1);-LearnedDepth(:)],1)...
               , opt);
  end
       solvesdp(F,norm( [Q1stXy; Q1stYy; Q1stYz; Q1stXz ;Q2ndXz;Q2ndYy;Q2ndYz;Q2ndXy;...
                    spdiags(AssociateM,[0],NuPatch,NuPatch)]...
                    *predictedM+[sparse(Q1size+Q2size,1);-LearnedDepth(:)],1)...
               , opt);
%                    spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)]...
    end
    predictedM = double(predictedM);
end

  % end        

if false
         cvx_begin
             cvx_quiet(false);
             variable predictedM(NuPatch,1);
%             variable kt(Q2size,1);
%             variable al(Q2size,1);
%             variable ga(Q2size,1);
             variable st(1,1);
%             minimize(st+norm(predictedM(cell2mat(LineIndex(:,1)'))-predictedM(cell2mat(LineIndex(:,2)')),1)+...
%                      norm(predictedM(cell2mat(LineIndexVert(:,3)'))-predictedM(cell2mat(LineIndexVert(:,4)')),1));
%              norm([Q2ndLine;Q2ndVertLine;...
             minimize(st);
              norm([Q1stXy;Q1stYy;Q2ndYz;Q2ndXz;...
                    spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)]...
                    *predictedM+[sparse(Q1size+Q2size,1);-LearnedDepth(:)])<=st;
%                    *predictedM+[sparse(SLsize,1);-LearnedDepth(:)])<=st;
%  	     [[Q2ndXz] -spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
%                spdiags(ones(Q2size,1),[0],Q2size,Q2size) sparse(Q2size,Q2size);...
%               [Q2ndXz] spdiags(ones(Q2size,1),[0],Q2size,Q2size)...
%                sparse(Q2size,Q2size) -spdiags(ones(Q2size,1),[0],Q2size,Q2size)]*[predictedM; kt; ga; al] == 0;
              
%             for q = 0:min(1,size(LineIndex,1)-1)
%	             predictedM(LineIndex{end-q,1})-predictedM(LineIndex{end-q,2})==0;
%             end
             predictedM>=0;
%             ga>=0;
%             al>=0;
         cvx_end   
%         mean(Q2ndLine*predictedM)
%         mean(Q2ndVertLine*predictedM)
end
%         figure(1); plot(predictedM-LearnedDepth(:));
%         figure(2); plot(Q2ndLine*predictedM);
%         figure(3); plot(Q2ndLine*predictedM);
%             for q = 0:min(1,(size(LineIndex,1)-1))
%	             predictedM(LineIndex{end-q,1})-predictedM(LineIndex{end-q,2})
%             end
%         toc;
%predictedM =LearnedDepth(:);
% post prosessing on straight line

% second optimization
       
        Date =date;
        predictedM =reshape(predictedM,VertYNuDepth,[]);
        depthMap = predictedM;
        depthfile = strrep(filename{i},'img','depth_learned'); % the depth filename
       system(['mkdir ' ScratchDataFolder '/DepthMRF_' Name{m} '_' DepthDirectory 'Clean2L1/']);
       save([ScratchDataFolder '/DepthMRF_' Name{m} '_' DepthDirectory 'Clean2L1/' depthfile '.mat'],'depthMap');
i%        save([ScratchDataFolder '/_predicted_' DepthDirectory '/' depthfile '.mat'],'depthMap');
        clear depthMap;
        
%=====================================================
        % 2d to would 3d
        [Position3DPredicted] = im_cr2w_cr(predictedM,RayCenter);
    
        % generate new LowResImgIndexSuperpixelSep_deoffset
        %LowResImgIndexSuperpixelSep_deoffset = LowResImgIndexSuperpixelSep{i};
    
        % add on image feature
        %global Imf;
        %Imf= cat(1,Position3DPredicted,permute(ones(VertY,1)*[1:HoriX],[3 1 2]), permute([1:VertY]'*ones(1,HoriX),[3 1 2]),permute(double(zeros(VertY,HoriX,3)),[3 1 2]),permute(double(LowResImgIndexSuperpixelSep_deoffset),[3 1 2]),permute(double(LowResImgIndexSuperpixelSep{i}),[3 1 2]) );
    
        % calculate each plane parameter for each superpixel
        %[PlaneParameterPredicted] = fit_all_planes(RayLoResCorner); % hard work around 2min
    
        % generate VRML
        Date = date;
%        [VrmlName] = vrml_test_faceset_triangle_nosky_v3(filename{i},Position3DPredicted,predictedM,RayCenter,['PredictM_' DepthDirectory '_' num2str(logScale) '_' num2str(SkyExclude) '_' Date '_LearnedLogScaleNonsky_PredictNonsky_L21st_L12ndSmooth_StraightLineDoblueStitchOrientEst_gravity'],a,b,Ox,Oy,maskSky{i},1,10);
%        size(VBrokeBook)
%        [VrmlName] = vrml_test_faceset_grid_nojump(filename{i},Position3DPredicted,predictedM,RayCenter,['PredictM_' DepthDirectory '_' num2str(logScale) '_' num2str(SkyExclude) '_' Date '_LearnedLogScaleNonsky_PredictNonsky_L21st_L12ndSmooth_StraightLineDoblueStitchOrientEst_gravity_nonSep'],HBrokeBook,VBrokeBook,0,maskSky{i},1,10,a,b,Ox,Oy,100);
      if m == 2
        [VrmlName] = vrml_test_faceset_goodSkyBoundary(  filename{i}, Position3DPredicted,predictedM,RayCenter, [Name{m} DepthDirectory], ...
                    [], [], 0, maskSky{i}, maskg{i}, 1, 0, a_default, b_default, Ox_default, Oy_default);
%        [VrmlName] = vrml_test_faceset_goodSkyBoundary(fiLENAME{i},Position3DPREDicted,predictedM,RayCenter,[Name{i} DepthDirectory],[],[],0,maskSky{i},1,0,a,b,Ox,Oy);
        system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
        %delete([ScratchDataFolder '/vrml/' VrmlName]);
      end
            %vrml_test_faceset_triangle(filename{i},PlaneParameterPredicted,LowResImgIndexSuperpixelSep{i},LowResImgIndexSuperpixelSep_deoffset,[DepthDirectory '_' Date]);

  end  % end of step 
end    
