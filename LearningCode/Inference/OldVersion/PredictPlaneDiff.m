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
function []=PredictPlane(DepthDirectory,logScale)
%this function generate the predicted plane
% It is the depth MRF


% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

% load data
load([ScratchDataFolder '/data/LowResImgIndexSuperpixelSep.mat']); % load LowResImgIndexSuperpixelSep
load([ScratchDataFolder '/data/DiffLowResImgIndexSuperpixelSep.mat']); % load DiffLowResImgIndexSuperpixelSep(medi$large)
load([ScratchDataFolder '/data/TextLowResImgIndexSuperpixelSep.mat']); % load TextLowResImgIndexSuperpixelSep using Textrea
load([ScratchDataFolder '/data/MaskGSky.mat']); % load maskg maskSky from CMU's output
load([ScratchDataFolder '/data/maskO.mat']);
% load useful features
load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixe

% prepare to store the predictedM
system(['mkdir ' ScratchDataFolder '/_predicted_' DepthDirectory]);

% set parameter
ZTiltFactor = 1; % both can be estimated after group fit that estimate the Norm_floor
YTiltFactor = 1;
% initial parameter
NuPics = size(filename,2);
for i = [10]%1 : NuPics
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
            load([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
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
            load([ScratchDataFolder '/_LearnDLogScale_' DepthDirectory '/' depthfile '.mat']);
        else
            load([ScratchDataFolder '/_LearnD_' DepthDirectory '/' depthfile '.mat']);
        end
            
        LearnedDepth = depthMap; clear depthMap;
        
        % initialize parameter
        NuPatch = VertYNuDepth*HoriXNuDepth;
        
        % generate specific ray for whole pics
        RayCenter = GenerateRay(HoriXNuDepth,VertYNuDepth,'center',a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
        Rx = RayCenter(:,:,1);
        Rx = Rx(:);
        Ry = RayCenter(:,:,2);
        Ry = Ry(:);
        Rz = RayCenter(:,:,3);
        Rz = Rz(:); 
        %RayCorner = GenerateRay(HoriXNuDepth,VertYNuDepth,'corner',a,b,Ox,Oy); %[ horiXSizeLowREs+1 VertYSizeLowREs+1 3] 
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
            [SecXG(j,:) SecYG(j,:)]= gen_2ndSmooth(NewEstGSup);
            [SecXV(j,:) SecYV(j,:)]= gen_2ndSmooth(NewEstVSup);
            [SecXS(j,:) SecYS(j,:)]= gen_2ndSmooth(NewEstSSup);

            % 1st order smooth
            [FirstYG(j,:) FirstXG(j,:)] = gen_1stSmooth(NewEstGSup);
            [FirstYV(j,:) FirstXV(j,:)] = gen_1stSmooth(NewEstVSup);
            [FirstYS(j,:) FirstXS(j,:)] = gen_1stSmooth(NewEstSSup);
            	%[GPy{j} ] = gen_GravityP_vertical(maskV);
            %[PlanePriorX PlanePriorY]= gen_PlanePrior(LowResImgIndexSuperpixelSep{i,1});
        end 
 
        % set weight for different segmentation
        small=50; med=25; large=5;
        temp =[small; med; large; small*ones(6,1); med*ones(6,1) ;large*ones(6,1)]
        Wei2ndSmoothGX = temp;
        Wei2ndSmoothGY = temp;
        Wei2ndSmoothVX = temp;
        Wei2ndSmoothVY = temp;
        Wei2ndSmoothSX = temp;
        Wei2ndSmoothSY = temp;
        Wei1stSmoothGX = temp;
        Wei1stSmoothGY = temp;
        Wei1stSmoothVX = temp;
        Wei1stSmoothVY = temp;
        Wei1stSmoothSX = temp;
        Wei1stSmoothSY = temp;

        % generate the smooth matrix
        M2ndSmoothY = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],[-1 0 1],NuPatch,NuPatch);
        M2ndSmoothX = spdiags([ones(NuPatch,1) -2*ones(NuPatch,1) ones(NuPatch,1)],...
                              [-VertYNuDepth 0 VertYNuDepth],NuPatch,NuPatch);
        M1stSmoothY = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 1],NuPatch,NuPatch);
        M1stSmoothX = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 VertYNuDepth],NuPatch,NuPatch);

        % generate beta
        beta2ndSmoothGX = Wei2ndSmoothGX'*SecXG;
        beta2ndSmoothGY = Wei2ndSmoothGY'*SecYG;
        beta2ndSmoothVX = Wei2ndSmoothVX'*SecXV;
        beta2ndSmoothVY = Wei2ndSmoothVY'*SecYV;
        beta2ndSmoothSX = Wei2ndSmoothSX'*SecXS;
        beta2ndSmoothSY = Wei2ndSmoothSY'*SecYS;
        beta1stSmoothGY = Wei2ndSmoothGY'*FirstYG;
        beta1stSmoothVY = Wei2ndSmoothVY'*FirstYV;
        beta1stSmoothSY = Wei2ndSmoothSY'*FirstYS;
        beta1stSmoothGX = Wei2ndSmoothGX'*FirstXG;
        beta1stSmoothVX = Wei2ndSmoothVX'*FirstXV;
        beta1stSmoothSX = Wei2ndSmoothSX'*FirstXS;
        
        % generate Q
        Q2ndXx = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch);
        Q2ndXy = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch);
        Q2ndXz = spdiags((beta2ndSmoothGX+beta2ndSmoothVX+beta2ndSmoothSX)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch);  
        Q2ndYx = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch);    
        Q2ndYy = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
        Q2ndYz = spdiags((beta2ndSmoothGY+beta2ndSmoothVY+beta2ndSmoothSY)'...
                 ,0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
        
        Q1stYz = spdiags((beta1stSmoothVY+beta1stSmoothSY)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
        Q1stYy = spdiags((beta1stSmoothGY)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothY*spdiags(Ry*YTiltFactor,0,NuPatch,NuPatch);
        Q1stXz = spdiags((beta1stSmoothGX+beta1stSmoothVX+beta1stSmoothSX)'...
                 ,0,NuPatch,NuPatch)*M1stSmoothX*spdiags(Rz*ZTiltFactor,0,NuPatch,NuPatch);
%         Q2ndGXx = spdiags(beta2ndSmoothGX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndGXy = spdiags(beta2ndSmoothGX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndGXz = spdiags(beta2ndSmoothGX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch); 
%         Q2ndVXx = spdiags(beta2ndSmoothVX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndVXy = spdiags(beta2ndSmoothVX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndVXz = spdiags(beta2ndSmoothVX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch); 
%         Q2ndSXx = spdiags(beta2ndSmoothSX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndSXy = spdiags(beta2ndSmoothSX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndSXz = spdiags(beta2ndSmoothSX',0,NuPatch,NuPatch)*M2ndSmoothX*spdiags(Rz,0,NuPatch,NuPatch); 
%        
%         Q2ndGYx = spdiags(beta2ndSmoothGY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndGYy = spdiags(beta2ndSmoothGY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndGYz = spdiags(beta2ndSmoothGY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
%         Q2ndVYx = spdiags(beta2ndSmoothVY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndVYy = spdiags(beta2ndSmoothVY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndVYz = spdiags(beta2ndSmoothVY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
%         Q2ndSYx = spdiags(beta2ndSmoothSY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rx,0,NuPatch,NuPatch); 
%         Q2ndSYy = spdiags(beta2ndSmoothSY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Ry,0,NuPatch,NuPatch); 
%         Q2ndSYz = spdiags(beta2ndSmoothSY',0,NuPatch,NuPatch)*M2ndSmoothY*spdiags(Rz,0,NuPatch,NuPatch); 
% ============================START MRF OPTMIZATIOM=========================================================
         %=================
         % generate mask for depth difference
         YDiff = repmat(logical([ones(VertYNuDepth-1,1); 0]),1,HoriXNuDepth);
         XDiff = repmat(([ones(1,HoriXNuDepth-1) 0]),VertYNuDepth,1);
         DMatrixY = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 1],NuPatch,NuPatch);
         DMatrixY = DMatrixY(~maskO,:);
         DMatrixX = spdiags([ones(NuPatch,1) -ones(NuPatch,1)],[0 VertYNuDepth],NuPatch,NuPatch);
         %=================
         tic;
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
%         predictedM = spdiags([ones(1,NuPatch) sparse(1,size(B,1)+size(Q,1))]',0,NuPatch...
%                             ,NuPatch+size(B,1)+size(Q,1))*x; 
        %predictedM = (LearnedDepth(:));
         cvx_begin
             cvx_quiet(false);
             variable predictedM(NuPatch,1);
             minimize(norm(DMatrixY*(predictedM - LearnedDepth(:)),1)...
             +norm((predictedM(maskO) - LearnedDepth(maskO)),1)...
             +norm([Q2ndXz;Q2ndYz]*predictedM)...
             +norm([Q1stXz]*predictedM));
             
             %+norm((beta1stSmoothGY+beta1stSmoothGX)'.*(predictedM - LearnedDepth(:)))...
             %+norm([Q1stYy ;Q1stYz; Q1stXz]*predictedM));
             %minimize(norm(predictedM - LearnedDepth(:))...
%              +norm([Q1stYy ;Q1stYz; Q1stXz]*predictedM)...
%              +norm([Q2ndXz;Q2ndYz]*predictedM));%...  % 2nd smooth Ground
%              %+norm([Q1stYz]*predictedM));
% %              +norm([Q2ndXz;Q2ndVYz]*predictedM,1)...  % vertical
% %              +norm([Q2ndSXz;Q2ndSYz]*predictedM,1));    % Sky
% %              +norm([Q2ndGXx;Q2ndGXy;Q2ndGXz;Q2ndGYx;Q2ndGYy;Q2ndGYz]*predictedM)...  % 2nd smooth Ground
% %              +norm([Q2ndVXx;Q2ndVXy;Q2ndVXz;Q2ndVYx;Q2ndVYy;Q2ndVYz]*predictedM)...  % vertical
% %              +norm([Q2ndSXx;Q2ndSXy;Q2ndSXz;Q2ndSYx;Q2ndSYy;Q2ndSYz]*predictedM));    % Sky
% %            + 100*norm(IPx{1}*(predictedM),1)...%+ 10*norm(IPx{2}*(predictedM.*Ry))+ 10*norm(IPx{2}*(predictedM.*Rz))..
% %            + 100*norm(IPy{1}*(predictedM),1)...%+ 10*norm(IPy{2}*(predictedM.*Ry))+ 10*norm(IPy{2}*(predictedM.*Rz))...
% %            + 50*norm(IPx{2}*(predictedM),1)...%+ 10*norm(IPx{2}*(predictedM.*Ry))+ 10*norm(IPx{2}*(predictedM.*Rz))..
% %            + 50*norm(IPy{2}*(predictedM),1)...%+ 10*norm(IPy{2}*(predictedM.*Ry))+ 10*norm(IPy{2}*(predictedM.*Rz))...
% %            + 5000*norm(GPy{5}*(predictedM.*Rz))...
% %            + 5000*norm(spdiags([ones(GSize,1) -ones(GSize,1)],[0 1],GSize-1,GSize)*(predictedM(maskg).*Ry(maskg)))...
% %            + 5000*norm(spdiags([ones(SkySize,1) -ones(SkySize,1)],[0 1],SkySize-1,SkySize)*(predictedM(masksky).*Rz(masksky))));
% %                  %+ 5000*norm(Gpy*(predictedM.*Ry))...
%                  %+ 5000*norm(spdiags([ones(SkySize,1) -1*ones(SkySize,1)],[0 1],SkySize-1,SkySize)*(predictedM(masksky).*Rz(masksky))));
%             %spdiags([ones(GSize,1) -ones(GSize,1)],[0 1],GSize-1,GSize)*(predictedM(maskg).*Ry(maskg)) == 0;
%             %spdiags([ones(SkySize,1) -ones(SkySize,1)],[0 1],SkySize-1,SkySize)*(predictedM(masksky).*Rz(masksky)) == 0;
             predictedM>=5;
%             %predictedM<=81;
         cvx_end    
         toc;
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
        [VrmlName] = vrml_test_faceset_triangle(filename{i},Position3DPredicted,RayCenter,['PredictM_' DepthDirectory '_' num2str(logScale) '_' Date '_Diff'],a,b,Ox,Oy);
        system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
        delete([ScratchDataFolder '/vrml/' VrmlName]);
            %vrml_test_faceset_triangle(filename{i},PlaneParameterPredicted,LowResImgIndexSuperpixelSep{i},LowResImgIndexSuperpixelSep_deoffset,[DepthDirectory '_' Date]);
end    
