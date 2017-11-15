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
function []=PredictPlane(DepthDirectory,logScale,SkyExclude)
%this function generate the predicted plane


% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

% load data
load([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat']); % load LowResImgIndexSuperpixelSep
load([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat']); % load DiffLowResImgIndexSuperpixelSep(medi$large)
load([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat']); % load TextLowResImgIndexSuperpixelSep using Textrea
load([ScratchDataFolder '/data/MaskGSky.mat']); % load maskg maskSky from CMU's output
%load([ScratchDataFolder '/data/maskO.mat']);
% load useful features
load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixe

% prepare to store the predictedM
system(['mkdir ' ScratchDataFolder '/_predicted_' DepthDirectory]);

% set parameter
ZTiltFactor = 1; % both can be estimated after group fit that estimate the Norm_floor
YTiltFactor = 1;
% initial parameter
NuPics = size(filename,2);
for i = 1 : NuPics
        i 
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
            eload([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
        end
    
        % load useful feature
        BatchNumber = ceil(i/batchSize);
        PicsNumber = mod(i,batchSize);
        if PicsNumber ==0
         PicsNumber = 10;
        end
%         load([ScratchDataFolder '/data/feature_sqrt_H4_ray' int2str(BatchNumber) '.mat']); % 'f'
%         f = f{PicsNumber};
%         f = f(:,1);
%         fsup = FeatureSuperpixel{i};
%         f = (fsup(1,f))'; 
        
        % load depthMap
        depthfile = strrep(filename{i},'img','depth_learned'); % the depth filename
        if logScale == 1
            if SkyExclude ==1
                load([ScratchDataFolder '/_LearnDLogScaleNonsky_' DepthDirectory '/' depthfile '.mat']);
            else
                load([ScratchDataFolder '/_LearnDLogScale_' DepthDirectory '/' depthfile '.mat']);
            end
        else
            if SkyExclude ==1
                load([ScratchDataFolder '/_LearnD_' DepthDirectory '/' depthfile '.mat']);
            else
                load([ScratchDataFolder '/_LearnD_' DepthDirectory '/' depthfile '.mat']);
            end
        end
            
        LearnedDepth = depthMap; clear depthMap;
        
        % initialize parameter
        NuPatch = VertYNuDepth*HoriXNuDepth;
        
        % generate straight line for straight line prior
        Img = imread([GeneralDataFolder '/' ImgFolder '/' filename{i} '.jpg'],'jpg');
        if  prod(size(Img))> SegVertYSize*SegHoriXSize*3
            Img = imresize(Img,[SegVertYSize SegHoriXSize],'nearest');
        end
        [ImgYSize ImgXSize dummy] = size(Img);
        [lineSegList] = edgeSegDetection(Img);
        NuSeg = size(lineSegList,1);
        % order the line segment by their length
        lineSegList = lineSegList./repmat([ImgXSize ImgYSize ImgXSize ImgYSize],NuSeg,1)...
                      .*repmat([HoriXNuDepth VertYNuDepth HoriXNuDepth VertYNuDepth],NuSeg,1);

        % take out the vertical edge so that we may combine it with 1stSmooth in Y direction
        VertLineSegList = lineSegList(abs(round(lineSegList(:,1))-round(lineSegList(:,3))) <=0 & ...
                                      abs(round(lineSegList(:,2))-round(lineSegList(:,4))) > 0 ,:);
        VertLineSegList = round(VertLineSegList);
        NuVertSeg = size(VertLineSegList,1);
        VertLineMask = zeros(VertYNuDepth, HoriXNuDepth);
        for k = 1:NuVertSeg
            if VertLineSegList(k,2) < VertLineSegList(k,4)
               VertLineMask(max(VertLineSegList(k,2),1):min((VertLineSegList(k,4)-1),VertYNuDepth),...
                            min(max(VertLineSegList(k,1),1),HoriXNuDepth)) = 1;
            else
               VertLineMask(max(VertLineSegList(k,4),1):min((VertLineSegList(k,2)-1),VertYNuDepth),...
                            min(max(VertLineSegList(k,1),1),HoriXNuDepth)) = 1;
            end
        end
         
        [StraightLinePriorSelectMatrix RayPorjectImgMapY LineIndex newd] = StraightLinePrior(lineSegList,LearnedDepth,a,b,Ox,Oy);
%         
        % generate ray porjection position in the image plane
        RayPorjectImgMapY = ((VertYNuDepth+1-RayPorjectImgMapY)-0.5)/VertYNuDepth - Oy; 
        RayPorjectImgMapX = repmat(((1:HoriXNuDepth)-0.5)/HoriXNuDepth - Ox,[VertYNuDepth 1]);
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
            sup(maskSky{i} == 1) = 0; 
            [SelectY{j} SelecrX{j} BM{j} Ns{j} MediaM{j} MeanM{j}] = OrientEst(sup, RayCenter);
        end 
 
        % set weight for different segmentation
        small=10; med=5; large=1;
        temp =[small; med; large; small*ones(6,1); med*ones(6,1) ;large*ones(6,1)]
        
        % set weight for the VertLineSeg Ground and VertialWalls
        WeiLinSegVertWall = 112;
        WeiLinSegVertGround = 112;

        % set weight for the straight line prior
        WeiStraightLine = 1120

        % generate the smooth matrix
        M2ndSmoothY = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],[-1 0 1],NuPatch,NuPatch);
        M2ndSmoothX = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],...
                              [-VertYNuDepth 0 VertYNuDepth],NuPatch,NuPatch);
        M1stSmoothY = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 1],NuPatch,NuPatch);
        M1stSmoothX = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 VertYNuDepth],NuPatch,NuPatch);

        % generate beta
%        beta2ndSmoothGX = Wei2ndSmoothGX'*ones(21,1) * logisticResponse(Wei2ndSmoothGX'*SecXG / (Wei2ndSmoothGX'*ones(21,1)) );
        beta2ndSmoothGX = Wei2ndSmoothGX'*SecXG;
%MIN: TO DO
%MIN: TO DO
        beta2ndSmoothGY = Wei2ndSmoothGY'*SecYG;
%        beta2ndSmoothGY = Wei2ndSmoothGY'*ones(21,1) * logisticResponse(Wei2ndSmoothGY'*SecYG / (Wei2ndSmoothGY'*ones(21,1)) );
        beta2ndSmoothGY = Wei2ndSmoothGY'*SecYG;
%        beta2ndSmoothVX = Wei2ndSmoothVX'*ones(21,1) * logisticResponse(Wei2ndSmoothVX'*SecXV / (Wei2ndSmoothVX'*ones(21,1)) );
        beta2ndSmoothVX = Wei2ndSmoothVX'*SecXV;
%        beta2ndSmoothVY = Wei2ndSmoothVY'*ones(21,1) * logisticResponse(Wei2ndSmoothVY'*SecYV / (Wei2ndSmoothVY'*ones(21,1)) );
        beta2ndSmoothVY = Wei2ndSmoothVY'*SecYV;
%        beta2ndSmoothSX = Wei2ndSmoothSX'*ones(21,1) * logisticResponse(Wei2ndSmoothSX'*SecXS / (Wei2ndSmoothSX'*ones(21,1)) );
%        beta2ndSmoothSY = Wei2ndSmoothSY'*ones(21,1) * logisticResponse(Wei2ndSmoothSY'*SecYS / (Wei2ndSmoothSY'*ones(21,1)) );

%        beta1stSmoothGY = Wei1stSmoothGY'*ones(21,1)*logisticResponse(Wei1stSmoothGY'*FirstYG/(Wei1stSmoothGY'*ones(21,1)) );
        beta1stSmoothGY = Wei2ndSmoothGY'*FirstYG;
%        beta1stSmoothVY = Wei1stSmoothVY'*ones(21,1)*logisticResponse(Wei1stSmoothVY'*FirstYV/(Wei1stSmoothVY'*ones(21,1)) );
        beta1stSmoothVY = Wei2ndSmoothVY'*FirstYV;
%        beta1stSmoothSY = Wei1stSmoothSY'*ones(21,1)*logisticResponse(Wei1stSmoothSY'*FirstYS/(Wei1stSmoothSY'*ones(21,1)) );
%        beta1stSmoothGX = Wei1stSmoothGX'*ones(21,1)*logisticResponse(Wei1stSmoothGX'*FirstXG/(Wei1stSmoothGX'*ones(21,1)) );
        beta1stSmoothGX = Wei2ndSmoothGX'*FirstXG;
%        beta1stSmoothVX = Wei1stSmoothVX'*ones(21,1)*logisticResponse(Wei1stSmoothVX'*FirstXV/(Wei1stSmoothVX'*ones(21,1)) );
        beta1stSmoothVX = Wei2ndSmoothVX'*FirstXV;
%        beta1stSmoothSX = Wei1stSmoothSX'*ones(21,1)*logisticResponse(Wei1stSmoothSX'*FirstXS/(Wei1stSmoothSX'*ones(21,1)) );
        
        Beta1stSmoothLineGY = VertLineMask.*(maskEstGVS==1)*WeiLinSegVertGround;
        Beta1stSmoothLineGY = (Beta1stSmoothLineGY(:))';
        Beta1stSmoothLineVWY = VertLineMask.*(maskEstGVS==2)*WeiLinSegVertWall;
        Beta1stSmoothLineVWY = (Beta1stSmoothLineVWY(:))';
        
        % generate Q
%        Q2ndXx = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch);
%        Q2ndXy = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch);
%        Q2ndXz = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch);  
%        Q2ndYx = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch);    
%        Q2ndYy = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
%        Q2ndYz = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
%                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
 
        Q2ndXx = spdiags((beta2ndSmoothGX+beta2ndSmoothVX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch);
        Q2ndXy = spdiags((beta2ndSmoothGX+beta2ndSmoothVX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch);
        %Q2ndXz = spdiags((beta2ndSmoothGX+beta2ndSmoothVX)'...
        %         ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch);  
        Q2ndXz = spdiags((beta2ndSmoothVX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch);  
        Q2ndYx = spdiags((beta2ndSmoothGY+beta2ndSmoothVY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch);    
        Q2ndYy = spdiags((beta2ndSmoothGY+beta2ndSmoothVY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
        %Q2ndYz = spdiags((beta2ndSmoothGY+beta2ndSmoothVY)'...
        %         ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
        Q2ndYz = spdiags((beta2ndSmoothVY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 

        Q2ndLinex = WeiStraightLine*StraightLinePriorSelectMatrix*spdiags(Rx,0,NuPatch,NuPatch);
        Q2ndLiney = WeiStraightLine*StraightLinePriorSelectMatrix*spdiags(Ry,0,NuPatch,NuPatch);
        Q2ndLinez = WeiStraightLine*StraightLinePriorSelectMatrix*spdiags(Rz,0,NuPatch,NuPatch);
         
%        Q1stYz = spdiags((beta1stSmoothVY+beta1stSmoothSY+Beta1stSmoothLineVWY)'...
%                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
        %Q1stYz = spdiags((Beta1stSmoothLineVWY)'...
        %         ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
%        Q1stYy = spdiags((beta1stSmoothGY+Beta1stSmoothLineGY)'...
%                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Ry*YTiltFactor,0,NuPatch,NuPatch);
%        Q1stXz = spdiags((beta1stSmoothGX+beta1stSmoothVX+beta1stSmoothSX)'...
%                 ,0,NuPatch,NuPatch)*M1stSmoothX*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);


        %Q1stYz = spdiags((beta1stSmoothVY+Beta1stSmoothLineVWY)'...
        %         ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
        Q1stYz = spdiags((beta1stSmoothVY+Beta1stSmoothLineVWY)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
        Q1stYy = spdiags((beta1stSmoothGY+Beta1stSmoothLineGY)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Ry*YTiltFactor,0,NuPatch,NuPatch);
        Q1stXz = spdiags((beta1stSmoothGX+beta1stSmoothVX)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothX*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
% ============================START MRF OPTMIZATIOM=========================================================
         %=================
         % generate mask for depth difference
         YDiff = repmat(logical([ones(VertYNuDepth-1,1); 0]),1,HoriXNuDepth);
         XDiff = repmat(([ones(1,HoriXNuDepth-1) 0]),VertYNuDepth,1);
         DMatrixY = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 1],NuPatch,NuPatch);
         %DMatrixY = DMatrixY(~maskO,:);
         DMatrixX = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 VertYNuDepth],NuPatch,NuPatch);
         %=================
         tic;
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
        Q1size = size([Q1stYy; Q1stYz;Q1stXz],1)
        SLsize = size(Q2ndLinez,1)
        Bsize = size([Q2ndXz;Q2ndYz],1)
%	E = [Q1stYy ;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);...
%              Q2ndLinez];
%        e = [sparse(Q1size,1); -LearnedDepth(:); sparse(SLsize,1)];
%        B = [Q2ndXz ; Q2ndYz];
%	H = 2*[[Q1stYy;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);Q2ndLinez]'*...
%             [Q1stYy;Q1stYz; Q1stXz;...
%              spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch);Q2ndLinez] sparse(NuPatch,Bsize*3);...
%              sparse(Bsize*3,NuPatch+Bsize*3)];
%        f =[2*[sparse(Q1size,1); -LearnedDepth(:); sparse(SLsize,1)]'*...
%            [Q1stYy;Q1stYz; Q1stXz...
%             spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch); Q2ndLinez] ones(1,Bsize) sparse(1,Bsize*2)]';
%        Aeq = [[Q2ndXz ; Q2ndYz] -spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
%               spdiags(ones(Bsize,1),[0],Bsize,Bsize) sparse(Bsize,Bsize);...
%              [Q2ndXz ; Q2ndYz] spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
%               sparse(Bsize,Bsize) spdiags(ones(Bsize,1),[0],Bsize,Bsize)];
%        beq = sparse(Bsize*2,1);
%        lb = [5*ones(NuPatch,1);sparse(Bsize*2,1);...
%              -2*max(LearnedDepth(:))*ones(Bsize,1)];
%        ub = [max(LearnedDepth(:))*ones(NuPatch,1);...
%              2*max(LearnedDepth(:))*ones(Bsize*2,1); sparse(Bsize,1)];
%        options = optimset('LargeScale', 'on', 'Display', 'off');
%        x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
%        predictedM = [spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)...
%                      sparse(NuPatch,Bsize*3)]*x;
%        A = [sparse(Bsize*2,NuPatch+Bsize)...
%             spdiags([-ones(Bsize,1);ones(Bsize,1)],[0],Bsize*2,Bsize*2)];
%        b =sparse(Bsize*2,1);
%=====================================================
%         predictedM = spdiags([ones(1,NuPatch) sparse(1,size(B,1)+size(Q,1))]',0,NuPatch...
%                             ,NuPatch+size(B,1)+size(Q,1))*x; 
%        predictedM = (LearnedDepth(:));


  	     %[[Q2ndYz] -spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
             %   spdiags(ones(Bsize,1),[0],Bsize,Bsize) sparse(Bsize,Bsize);...
             %  [Q2ndYz] spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
             %   sparse(Bsize,Bsize) spdiags(ones(Bsize,1),[0],Bsize,Bsize)]*[predictedM; kt; ga; al] == 0;
              
%             [Q2ndXz; Q2ndYz]*predictedM <=kt;
%             [Q2ndXz; Q2ndYz]*predictedM >= -kt;
%             predictedM>=5;
              %ga>=0;
              %al<=0;
%         cvx_end   
%         toc;

% post prosessing on straight line

% second optimization
       
        Date =date;
%        predictedM =reshape(predictedM,VertYNuDepth,[]);
         % quadprog Start
%         LearnedDepth = single(LearnedDepth);
         cvx_begin
             cvx_quiet(false);
             variable predictedM(NuPatch,1);
             variable kt(Bsize,1);
             %variable al(Bsize,1);
             %variable ga(Bsize,1);
             variable st(1,1);
             minimize(st);%+ones(1,Bsize)*kt);
              norm([Q2ndYy; Q2ndYz; Q2ndXz; Q2ndLinez;...
                    spdiags(ones(NuPatch,1),[0],NuPatch,NuPatch)]...
                    *predictedM+[sparse(Q1size+SLsize,1); -LearnedDepth(:)])<=st;
              
  	     %[[Q2ndYz] -spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
             %   spdiags(ones(Bsize,1),[0],Bsize,Bsize) sparse(Bsize,Bsize);...
             %  [Q2ndYz] spdiags(ones(Bsize,1),[0],Bsize,Bsize)...
             %   sparse(Bsize,Bsize) spdiags(ones(Bsize,1),[0],Bsize,Bsize)]*[predictedM; kt; ga; al] == 0;
              
%             [Q2ndXz; Q2ndYz]*predictedM <=kt;
%             [Q2ndXz; Q2ndYz]*predictedM >= -kt;
             predictedM>=5;
              %ga>=0;
              %al<=0;
         cvx_end   
%         toc;

% post prosessing on straight line

% second optimization
       
        Date =date;
        predictedM =reshape(predictedM,VertYNuDepth,[]);
        depthMap = predictedM;
        save([ScratchDataFolder '/_predicted_' DepthDirectory '/' depthfile '_' num2str(logScale) '_' Date '.mat'],'depthMap');
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
        [VrmlName] = vrml_test_faceset_triangle(filename{i},Position3DPredicted,RayCenter,['PredictM_' DepthDirectory '_' num2str(logScale) '_' num2str(SkyExclude) '_' Date '_LearnedLogScaleNonsky_PredictNonsky_L21st_L12ndSmooth_StraightLineDoblueStitchOrientEst_gravity'],a,b,Ox,Oy);
        system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
        delete([ScratchDataFolder '/vrml/' VrmlName]);
            %vrml_test_faceset_triangle(filename{i},PlaneParameterPredicted,LowResImgIndexSuperpixelSep{i},LowResImgIndexSuperpixelSep_deoffset,[DepthDirectory '_' Date]);
end    
