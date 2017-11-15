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
function  ReportPlaneParaMRF_FreeSup(step, DepthFolder,...
          Sup,MedSup,depthMap,VarMap,RayOri, Ray,MedRay,maskSky,maskG,Algo,k, CornerList, OccluList,...
          MultiScaleSupTable, StraightLineTable, HBrokeBook, VBrokeBook,previoslyStored,...
          baseline);
% This function runs the RMF over the plane parameter of each superpixels

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

if nargin <20
   baseline = 0;
end

% initialize parameters
displayFlag = false;
RenderVrmlFlag = true;
StickHori = 5%0.1; % sticking power in horizontal direction
StickVert = 10;     % sticking power in vertical direction
Center = 5; % Co-Planar weight at the Center of each superpixel
HoriConf = 1; % set the confidant of the learned depth at the middle in Horizontal direction of the image
VertConf = 1; % set the confidant of the learned depth at the top of the image
mapVert = linspace(VertConf,1,VertYNuDepth); % modeling the gravity prior
mapHori = [linspace(HoriConf,1,round(HoriXNuDepth/2)) fliplr(linspace(HoriConf,1,HoriXNuDepth-round(HoriXNuDepth/2)))]; % modeling the gravity prior
% ========set the range of depth that our model in
ClosestDist = 1;
% set the FarestDist to very 5 times to median depth
FarestDist = 5*median(depthMap(:))
% ================================================
ceiling = 0*VertYNuDepth; % set the position of the ceiling, related to No plane coming back constrain
Name{1} = 'FraWOPri';
Name{2} = 'FraCoP';
%pause
if isempty(MultiScaleSupTable)
   Name{3} = 'Var_FraStickCoP';
else
   Name{3} = 'Var_FraStickCoPSTasCoP';
end
if ~isempty(MultiScaleSupTable)
   MultiScaleFlag = true;
   WeiV = 2*ones(1,size(MultiScaleSupTable,2)-1);
else
   MultiScaleFlag = false;
   WeiV = 1;
end
WeiV(1,1:2:end) = 6; % emphasize the middle scale three times smaller than large scale
WeiV =WeiV./sum(WeiV);% normalize if pair of superpixels have same index in all the scale, their weight will be 10
ShiftStick = -.1;  % between -1 and 0, more means more smoothing.
ShiftCoP = -.5;  % between -1 and 0, more means more smoothing.
gravity =true; % if true, apply the HoriConf and VertConf linear scale weight
CoPST = true; % if true, apply the Straight line prior as the Co-Planar constrain
ConerImprove = true;
FractionalDepthError = true;
% get rid of the error point and sky point in the depthMap
% set every depth bigger than FarestDistmeter to FarestDistmeters 
%CleanedDepthMap = (depthMapif ~previoslyStored >80).*medfilt2(depthMap,[4 4])+(depthMap<=80).*depthMap;
CleanedDepthMap = depthMap;
%CleanedDepthMap(depthMap>FarestDist) = FarestDist; % don't clean the point >80 sometimes it occlusion
disp('Nu of depthMap>FarestDist');
sum(sum(depthMap>FarestDist))
CleanedDepthMap(depthMap>FarestDist) = NaN; % don't clean the point >80 sometimes it occlusion
size(CleanedDepthMap)
size(depthMap)
Posi3D = im_cr2w_cr(CleanedDepthMap,permute(Ray,[2 3 1]));
if ~previoslyStored

NewMap = [rand(max(Sup(:)),3); [0 0 0]];
% Clean the Sup near sky
%maskSky = imdilate(maskSky, strel('disk', 3) );
%[Sup,MedSup]=CleanedSup(Sup,MedSup,maskSky);
maskSky = Sup == 0;
maskSkyEroded = imerode(maskSky, strel('disk', 4) );
SupEpand = ExpandSup2Sky(Sup,maskSkyEroded);
NuPatch = HoriXNuDepth*VertYNuDepth-sum(maskSky(:));

NuSup = setdiff(unique(Sup)',0);
NuSup = sort(NuSup);
NuSupSize = size(NuSup,2);

% Sup index and planeParameter index inverse map
Sup2Para = sparse(1,max(Sup(:)));
Sup2Para(NuSup) = 1:NuSupSize;

% constructinf the Straight line prior matrix Will be add in the CoPlane matrix
NuLine = size(StraightLineTable,2);
CoPSTList = [];
for i = 1:NuLine
    L = StraightLineTable{i};
    X = L(1:(end-1))';
    Y = L(2:end)';
    if isempty(X)
       continue;
    end
    for j = 1:size(X,1)
        if X(j)~=Y(j)
           CoPSTList = [CoPSTList; X(j) Y(j)];
        end
    end
end
end

% Generate the Matrix for MRF
tic
PosiM = sparse(0,0);
VarM = sparse(0,0);
RayMd = sparse(0,0);
RayAllOriM = sparse(0,0);
RayAllM = sparse(0,0);
RayMtilt = sparse(0,0);
RayMCent = sparse(0,0);
DepthInverseMCent = [];
DepthInverseM = [];
YPointer = [];
beta = [];
EmptyIndex = [];
for i = NuSup
%    mask = Sup ==i;
    mask = SupEpand ==i; % include the Ray that will be use to expand the NonSky
    RayAllOriM = blkdiag( RayAllOriM, RayOri(:,mask)');
    RayAllM = blkdiag( RayAllM, Ray(:,mask)');
    mask = Sup ==i; % Not include the Ray that will be use to expand the NonSky    
    [yt x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(yt));
    YPointer = [YPointer; CenterY >= ceiling];
    mask(isnan(CleanedDepthMap)) = false;
    SupNuPatch(i) = sum(mask(:));
%    if sum(mask(:)) < 5
%       EmptyIndex = [EmptyIndex; i];
%       mask(:) = false;
%    end
    % find center point
    [yt x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(yt));
  
  if ~all(mask(:)==0)
    if gravity
      Pa2CenterRatio = median(CleanedDepthMap(mask))./CleanedDepthMap(mask);
      if sum(mask(:)) > 0
         RayMtilt = blkdiag(RayMtilt, ...
             ( Pa2CenterRatio(:,[1 1 1]).*repmat(RayOri(:,CenterY,CenterX)',[ SupNuPatch(i) 1])- RayOri(:,mask)'));
      else
         RayMtilt = blkdiag(RayMtilt, RayOri(:,mask)');
      end
      RayMCent = blkdiag(RayMCent, RayOri(:,CenterY,CenterX)'*SupNuPatch(i)*mapVert(CenterY)*mapHori(CenterX));
      PosiM = blkdiag(PosiM,Posi3D(:,mask)'.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      VarM = [VarM; VarMap(mask).*(mapVert(yt)').*( mapHori(x)')];
      RayMd = blkdiag(RayMd,RayOri(:,mask)'.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      DepthInverseMCent = [DepthInverseMCent; 1./median(CleanedDepthMap(mask)).*SupNuPatch(i)* mapVert(CenterY)'*mapHori(CenterX)];
      DepthInverseM = [DepthInverseM; 1./CleanedDepthMap(mask).* mapVert(yt)'.*mapHori(x)'];
    else
      Pa2CenterRatio = CleanedDepthMap(CenterY,CenterX)./CleanedDepthMap(mask);
      if sum(mask(:)) > 0
         RayMtilt = blkdiag(RayMtilt, ...
             ( Pa2CenterRatio(:,[1 1 1]).*repmat(RayOri(:,CenterY,CenterX)',[ SupNuPatch(i) 1])- RayOri(:,mask)'));
      alse
         RayMtilt = blkdiag(RayMtilt, RayOri(:,mask)');
      end
      RayMCent = blkdiag(RayMCent, RayOri(:,CenterY,CenterX)'*SupNuPatch(i));
      PosiM = blkdiag(PosiM,Posi3D(:,mask)');
      VarM = [VarM; VarMap(mask)];
      RayMd = blkdiag(RayMd,RayOri(:,mask)');
      DepthInverseMCent = [DepthInverseMCent; 1./median(CleanedDepthMap(mask)).*SupNuPatch(i)];
      DepthInverseM = [DepthInverseM; 1./CleanedDepthMap(mask)];
    end
  else
     RayMtilt = blkdiag(RayMtilt, RayOri(:,mask)');
     RayMCent = blkdiag(RayMCent, RayOri(:,mask)');
     PosiM = blkdiag(PosiM, Posi3D(:,mask)');
     VarM = [VarM; VarMap(mask)];
     RayMd = blkdiag(RayMd, RayOri(:,mask)');
  end
end
YPointer(YPointer==0) = -1;

% buliding CoPlane Matrix=========================================================================
BounaryPHori = conv2(Sup,[1 -1],'same') ~=0;
BounaryPHori(:,end) = 0;
BounaryPVert = conv2(Sup,[1; -1],'same') ~=0;
BounaryPVert(end,:) = 0;
ClosestNList = [ Sup(find(BounaryPHori==1)) Sup(find(BounaryPHori==1)+VertYNuDepth);...
                 Sup(find(BounaryPVert==1)) Sup(find(BounaryPVert==1)+1)];
ClosestNList = sort(ClosestNList,2);
ClosestNList = unique(ClosestNList,'rows');
ClosestNList(ClosestNList(:,1) == 0,:) = [];
BoundaryAll = BounaryPHori + BounaryPHori(:,[1 1:(end-1)])...
             +BounaryPVert + BounaryPVert([1 1:(end-1)],:);
BoundaryAll([1 end],:) = 1;
BoundaryAll(:,[1 end]) = 1;
NuSTList = 0;
if CoPST
   ClosestNList = [ClosestNList; CoPSTList];
   NuSTList = size(CoPSTList,1)
end
NuNei = size(ClosestNList,1);
CoPM1 = sparse(0,3*NuSupSize);
CoPM2 = sparse(0,3*NuSupSize);
CoPEstDepth = sparse(0,0);
CoPNorM = [];
WeiCoP = [];
for i = 1: NuNei
%  if ~CornerList(i)
    mask = Sup == ClosestNList(i,1);
    SizeMaskAll = sum(mask(:));
    [y x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(y));
    y = find(mask(:,CenterX));
    if ~isempty(y)
       CenterY = round(median(y));
    end
%    CenterX = round(median(x));
%    CenterY = round(median(yt));
    
    temp1 = sparse(1, 3*NuSupSize);
    temp2 = sparse(1, 3*NuSupSize);
    temp1(:,(Sup2Para(ClosestNList(i,1))*3-2): Sup2Para(ClosestNList(i,1))*3) = Ray(:,CenterY,CenterX)';
    temp2(:,(Sup2Para(ClosestNList(i,2))*3-2): Sup2Para(ClosestNList(i,2))*3) = Ray(:,CenterY,CenterX)';
    NuNei-NuSTList
    i
    if i < NuNei-NuSTList % only immediate connecting superpixels are weighted using MultiScaleSup
%       wei = WeiV*10;%*(MultiScaleSupTable(Sup2Para(ClosestNList(i,1)),2:end) == MultiScaleSupTable(Sup2Para(ClosestNList(i,2)),2:end))';  
       if MultiScaleFlag 
          vector = (MultiScaleSupTable(Sup2Para(ClosestNList(i,1)),2:end) == MultiScaleSupTable(Sup2Para(ClosestNList(i,2)),2:end));
          expV = exp(-10*(WeiV*vector' + ShiftCoP) );
          wei = 1/(1+expV);
       else
          wei = 1;
       end
    else
       wei = 1;
    end
    CoPM1 = [CoPM1; temp1*wei];
    CoPM2 = [CoPM2; temp2*wei];
    tempWeiCoP = [SizeMaskAll];
    CoPEstDepth = [CoPEstDepth; max(median(CleanedDepthMap(mask)),ClosestDist)];
    
    mask = Sup == ClosestNList(i,2);
    SizeMaskAll = sum(mask(:));
    [y x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(y));
    y = find(mask(:,CenterX));
    if ~isempty(y)
       CenterY = round(median(y));
    end
%    [yt x] = find(mask);
%    CenterX = round(median(x));
%    CenterY = round(median(yt));

    temp1 = sparse(1, 3*NuSupSize);
    temp2 = sparse(1, 3*NuSupSize);
    temp1(:,(Sup2Para(ClosestNList(i,1))*3-2): Sup2Para(ClosestNList(i,1))*3) = Ray(:,CenterY,CenterX)';
    temp2(:,(Sup2Para(ClosestNList(i,2))*3-2): Sup2Para(ClosestNList(i,2))*3) = Ray(:,CenterY,CenterX)';
    CoPM1 = [CoPM1; temp1*wei];
    CoPM2 = [CoPM2; temp2*wei];
    tempWeiCoP = [tempWeiCoP; SizeMaskAll];
    WeiCoP = [WeiCoP; tempWeiCoP];
    CoPEstDepth = [CoPEstDepth; max(median(CleanedDepthMap(mask)),ClosestDist)];
%  end
end%=========================================================================================================

% find the boundary point that might need to be stick ot each other==========================================
HoriStickM_i = sparse(0,3*NuSupSize);
HoriStickM_j = sparse(0,3*NuSupSize);
HoriStickPointInd = [];
EstDepHoriStick = [];
for i = find(BounaryPHori==1)'
    j = i+VertYNuDepth;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
%  if ~OccluList(sum( ClosestNList == repmat(sort([Sup(i) Sup(j)]), [NuNei  1]),2) == 2)
%  size(OccluList)
%  if ~any(sum( OccluList == repmat(sort([Sup(i) Sup(j)]), [size(OccluList,1)  1]),2) == 2)
    Target(1) = Sup2Para(Sup(i));
    Target(2) = Sup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
%    betaTemp = StickHori;%*(DistStickLengthNormWei.^2)*beta(Target(I));
%    betaTemp = StickHori*WeiV;%*(MultiScaleSupTable(Sup2Para(Sup(i)),2:end) == MultiScaleSupTable(Sup2Para(Sup(j)),2:end))';%*(DistStickLengthNormWei.^2)*beta(Target(I));
    if MultiScaleFlag
          vector = (MultiScaleSupTable(Sup2Para(Sup(i)),2:end) == MultiScaleSupTable(Sup2Para(Sup(j)),2:end));
          expV = exp(-10*(WeiV*vector' + ShiftStick) );
       betaTemp = StickHori*(0.5+1/(1+expV)); %*(DistStickLengthNormWei.^2)*beta(Target(I)); 
       % therr should always be sticking (know don't care about occlusion)
    else
       betaTemp = StickHori;
    end
    temp = sparse(3,NuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    HoriStickM_i = [HoriStickM_i; betaTemp*temp(:)'];
    temp = sparse(3,NuSupSize);
    temp(:,Target(2)) = rayBoundary(:,2);
    HoriStickM_j = [HoriStickM_j; betaTemp*temp(:)'];
    EstDepHoriStick = [EstDepHoriStick; sqrt(max(CleanedDepthMap(i),ClosestDist)*max(CleanedDepthMap(j),ClosestDist))];
    HoriStickPointInd = [HoriStickPointInd i ];
%  else
%    disp('Occlu');
%  end
end
VertStickM_i = sparse(0,3*NuSupSize);
VertStickM_j = sparse(0,3*NuSupSize);
VertStickPointInd = [];
EstDepVertStick = [];
for i = find(BounaryPVert==1)'
    j = i+1;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
%  if ~OccluList(sum( ClosestNList == repmat(sort([Sup(i) Sup(j)]), [NuNei  1]),2) == 2)
%  if ~any(sum( OccluList == repmat(sort([Sup(i) Sup(j)]), [size(OccluList,1)  1]),2) == 2)
    Target(1) = Sup2Para(Sup(i));
    Target(2) = Sup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
%    betaTemp = StickVert;%*(DistStickLengthNormWei.^2)*beta(Target(I));
    if MultiScaleFlag
       vector = (MultiScaleSupTable(Sup2Para(Sup(i)),2:end) == MultiScaleSupTable(Sup2Para(Sup(j)),2:end));
       expV = exp(-10*(WeiV*vector' + ShiftStick) );
       betaTemp = StickVert*(0.5+1/(1+expV));
       % therr should always be sticking (know don't care about occlusion)
    else
       betaTemp = StickVert;
    end
    temp = sparse(3,NuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    VertStickM_i = [VertStickM_i; betaTemp*temp(:)'];
    temp = sparse(3,NuSupSize);
    temp(:,Target(2)) = rayBoundary(:,2);
    VertStickM_j = [VertStickM_j; betaTemp*temp(:)'];
    EstDepVertStick = [EstDepVertStick; sqrt(max(CleanedDepthMap(i),ClosestDist)*max(CleanedDepthMap(j),ClosestDist))];
    VertStickPointInd = [VertStickPointInd i ];
%  else
%    disp('Occlu');
%  end
end
% ======================================Finish building up matrix=====================hard work======================
% ================================================================================================================
depthfile = strrep(filename{k},'img','depth_learned'); %

for i = step
    opt = sdpsettings('solver','sedumi');
    ParaPPCP = sdpvar(3*NuSupSize,1);
    F = set(ParaPPCP(3*(1:NuSupSize)-1).*YPointer<=0) ...
        +set(RayAllM*ParaPPCP >=1/FarestDist)...
        +set(RayAllM*ParaPPCP <=1/ClosestDist)...
        +set(RayAllOriM*ParaPPCP >=1/FarestDist)...
        +set(RayAllOriM*ParaPPCP <=1/ClosestDist);
% First fit the plane to find the estimated plane parameters
% If PosiM contain NaN data the corresponding Plane Parameter will also be NaN
    if i == 1 % W/O  priors
    solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./(abs(VarM)),1)...
             , opt);
    elseif i ==2 % Coner
    solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./(abs(VarM)),1)...
             +Center*norm(((CoPM1 - CoPM2)*ParaPPCP).*CoPEstDepth, 1)...
             , opt);
    else % all the priors
    sum(sum(isnan(PosiM)))
    disp('Whole MRF')
    solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./(abs(VarM)),1)...
             +Center*norm(((CoPM1 - CoPM2)*ParaPPCP).*CoPEstDepth, 1)...
             +norm(((HoriStickM_i-HoriStickM_j)*ParaPPCP).*EstDepHoriStick,1)...
             +norm(((VertStickM_i-VertStickM_j)*ParaPPCP).*EstDepVertStick,1)...
             , opt);
    end

    ParaPPCP = double(ParaPPCP);
    isnan(ParaPPCP)
    SepPointMeasureHori = 1./(HoriStickM_i*ParaPPCP)-1./(HoriStickM_j*ParaPPCP);
    SepPointMeasureVert = 1./(VertStickM_i*ParaPPCP)-1./(VertStickM_j*ParaPPCP);
    ParaPPCP = reshape(ParaPPCP,3,[]);
    %any(any(isnan(ParaPPCP)))
    % porject the ray on planes to generate the ProjDepth
    FitDepthPPCP = FarestDist*ones(1,VertYNuDepth*HoriXNuDepth);
    FitDepthPPCP(~maskSkyEroded) = (1./sum(ParaPPCP(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
    FitDepthPPCP = reshape(FitDepthPPCP,VertYNuDepth,[]);
    % storage for comparison purpose ======================
    depthMap = FarestDist*ones(1,VertYNuDepth*HoriXNuDepth);
    depthMap(~maskSkyEroded) = (1./sum(ParaPPCP(:,Sup2Para(SupEpand(~maskSkyEroded))).*RayOri(:,~maskSkyEroded),1))';
    depthMap = reshape(depthMap,VertYNuDepth,[]);
%    if baseline == 0
       system(['mkdir ' ScratchDataFolder '/' Name{i} '_' DepthFolder '/']);
       save([ScratchDataFolder '/' Name{i} '_' DepthFolder '/' depthfile '.mat'],'depthMap');
%    elseif baseline == 1
%       system(['mkdir ' ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/']);
%       save([ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/' depthfile '.mat'],'depthMap');
%    else
%       system(['mkdir ' ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/']);
%       save([ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/' depthfile '.mat'],'depthMap');
%    end
  
    % =====================================================
    [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
    if false %RenderVrmlFlag && i==3
       [VrmlName] = vrml_test_faceset_goodSkyBoundary(	filename{k}, Position3DFitedPPCP, FitDepthPPCP, permute(Ray,[2 3 1]), [Name{i} DepthFolder], ...
                    [], [], 0, maskSkyEroded, maskG, 1, 0, a_default, b_default, Ox_default, Oy_default);
       system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
       delete([ScratchDataFolder '/vrml/' VrmlName]);
    end
%    save([ScratchDataFolder '/data/PreDepth/FitDepthPPCP' num2str(k) '.mat'],'FitDepthPPCP','SepPointMeasureHori',...
%      'SepPointMeasureVert','VertStickPointInd','HoriStickPointInd');
end

%return;
%==================Finished for one step MRF==========================================================================================================


% ====================================following are 2nd and 3rd step MRF to give more visually pleasing result=======================================
;% generating new PosiMPPCP using the new position

save([ScratchDataFolder '/data/VertShiftVrml' mat2str(k) '.mat' ] );
  groundThreshold = cos(15*pi/180);
  verticalThreshold = cos(50*pi/180); % before Nov 29 11:30Pm I used 30 which is too big
  normPara = norms(ParaPPCP);
  normalizedPara = ParaPPCP ./ repmat( normPara, [3 1]);
  groundPara = abs(normalizedPara(2,:)) >= groundThreshold;
  groundParaInd = find(groundPara);
  verticalPara = abs(normalizedPara(2,:)) <= verticalThreshold;
  verticalParaInd = find(verticalPara);

 if false
    [CoPEstDepth CoPM1 CoPM2 EstDepHoriStick HoriStickM_i HoriStickM_j EstDepVertStick VertStickM_i VertStickM_j WeiCoP NewSup NewNuSup NewNuSupSize NewSup2Para FixPara VertStickPointInd HoriStickPointInd] ...
            = FreeSupSharpCorner( Sup, Ray, OccluList, WeiV, ShiftStick, StickHori, StickVert, ClosestDist, depthMap,...
              MultiScaleSupTable, verticalParaInd, graoundParaInd, MultiScaleFlag);

 end

  indexVertical = find( verticalPara)*3-1;
  indexGroundX = find( groundPara)*3-2;
  indexGroundZ = find( groundPara)*3;

  PosiMPPCP = sparse(0,0);
  VarM2 = sparse(0,0);
%  VertVar = 10^(-2);
  for i = NuSup
      mask = Sup == i;
      if any(verticalParaInd == Sup2Para(i))
         mask = logical(zeros(size(Sup)));
      end
      mask(ipermute( isnan( sum(Posi3D,1) ),[2 3 1] ) ) = false; % new added to avoid NaN
%         PosiMPPCP = blkdiag(PosiMPPCP, Posi3D(:,mask)');
%         VarM2 = [VarM2; VertVar*ones(sum(mask(:)), 1)];         
  %       mask = logical(zeros(size(Sup)));
%      elseif any(groundParaInd == Sup2Para(i))
%         PosiMPPCP = blkdiag(PosiMPPCP, Position3DFitedPPCP(:,mask)');
%         VarM2 = [VarM2; VarMap(mask)];         
%      else
         PosiMPPCP = blkdiag(PosiMPPCP, Posi3D(:,mask)');
         VarM2 = [VarM2; VarMap(mask)];         
%      end
  end

  opt = sdpsettings('solver','sedumi');
  Para = sdpvar(3*NuSupSize,1);
  F = set(Para(3*(1:NuSupSize)-1).*YPointer<=0) + set(RayAllM*Para >=1/FarestDist)...
      +set(RayAllM*Para <=1/ClosestDist)...
      +set(Para(indexVertical) == 0);
%      +set(Para(indexGroundX) == 0)...
%      +set(Para(indexGroundZ) == 0);
if FractionalDepthError
   size(PosiMPPCP)
   solvesdp(F,norm( ( PosiMPPCP*Para-ones(size(PosiMPPCP,1),1))./VarM2,1)...
             +Center*norm(((CoPM1 - CoPM2)*Para).*CoPEstDepth, 1)...
             +norm(((HoriStickM_i-HoriStickM_j)*Para).*EstDepHoriStick,1)...
             +norm(((VertStickM_i-VertStickM_j)*Para).*EstDepVertStick,1)...
             , opt);
%             +1000*norm( Para(indexGroundX)./ normPara( find(groundPara) )', 1)  ...
%             +1000*norm( Para(indexGroundZ)./ normPara( find(groundPara) )', 1) ...
%             sqrt( max(CoPM1*ParaPPCP(:), 1/FarestDist).*max(CoPM2*ParaPPCP(:), 1/FarestDist)), 1)...
%             +norm(((VertStickM_i-VertStickM_j)*Para)./...
%             sqrt( max(VertStickM_i*ParaPPCP(:), 1/FarestDist).*max(VertStickM_j*ParaPPCP(:), 1/FarestDist)),1)...
%   pause;
    %         +1000*norm( Para(indexVertical)./ normPara( find(verticalPara) )',1) ...
%            +norm(((HoriStickM_i-HoriStickM_j)*Para)./sqrt((VertStickM_i*ParaPPCP(:)).*(VertStickM_j*ParaPPCP(:))),1)...
else
   solvesdp(F,norm( RayMd*Para - DepthInverseM,1)... 
            +Center*norm((CoPM1 - CoPM2)*Para,1)...
            +norm((VertStickM_i-VertStickM_j)*Para,1)...
            +norm((HoriStickM_i-HoriStickM_j)*Para,1)...
            +1000*norm( Para(indexVertical)./ normPara( find(verticalPara) )',1) ...
            +1000*norm( Para(indexGroundX)./ normPara( find(groundPara) )', 1)  ...
            +1000*norm( Para(indexGroundZ)./ normPara( find(groundPara) )', 1) ...
            , opt);
%         +0.01*norm( RayMtilt*ParaPPCP,1)...        
%         +norm(spdiags(WeiCoP,0,size(CoPM1,1),size(CoPM1,1))*(CoPM1 - CoPM2)*ParaPPCP,1)...
%         +norm( (CoPM1 - CoPM2)*ParaPPCP,1 )...
end
%save([ScratchDataFolder '/data/All.mat']);
  Para = double(Para);
  SepPointMeasureHori = 1./(HoriStickM_i*Para)-1./(HoriStickM_j*Para);
  SepPointMeasureVert = 1./(VertStickM_i*Para)-1./(VertStickM_j*Para);
  Para = reshape(Para,3,[]);

  FitDepth = 80*ones(1,VertYNuDepth*HoriXNuDepth);
  FitDepth(~maskSkyEroded) = (1./sum(Para(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
  FitDepth = reshape(FitDepth,VertYNuDepth,[]);
  sum(isnan(FitDepth(:)))
  [Position3DFited] = im_cr2w_cr(FitDepth,permute(Ray,[2 3 1]));
%  [VrmlName] = vrml_test_faceset_goodSkyBoundary(	filename{k}, Position3DFited, FitDepth, permute(Ray,[2 3 1]), 'VNonSupport', ...
%					[], [], 0, maskSkyEroded, maskG, 1, 0, a_default, b_default, Ox_default, Oy_default);
 % system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
 % delete([ScratchDataFolder '/vrml/' VrmlName]);

%end
% ============================================================
if ConerImprove
save([ScratchDataFolder '/data/All.mat']);

  groundThreshold = cos(5*pi/180); % a little bigger than 15
  verticalThreshold = cos(50*pi/180);
  normPara = norms(Para);
  normalizedPara = Para ./ repmat( normPara, [3 1]);
  groundPara = abs(normalizedPara(2,:)) >= groundThreshold;
  groundParaInd = find(groundPara);
  verticalPara = abs(normalizedPara(2,:)) <= verticalThreshold;
  verticalParaInd = find(verticalPara);
 [CoPEstDepth CoPM1 CoPM2 EstDepHoriStick HoriStickM_i HoriStickM_j EstDepVertStick VertStickM_i VertStickM_j ...
                 WeiCoP NewSup NewNuSup NewNuSupSize NewSup2Para FixPara VertStickPointInd HoriStickPointInd] ...
            = FreeSupSharpCorner( Sup, Ray, OccluList, WeiV, ShiftStick, StickHori, StickVert, ClosestDist, depthMap,...
              MultiScaleSupTable, verticalParaInd, groundParaInd, MultiScaleFlag, Sup2Para);

%    [CoPM1 CoPM2 HoriStickM_i HoriStickM_j VertStickM_i VertStickM_j WeiCoP NewSup NewNuSup NewNuSupSize NewSup2Para FixPara VertStickPointInd HoriStickPointInd] ...
%            = FreeSupSharpCorner( Sup, Ray, OccluList, WeiV, ShiftStick, StickHori, StickVert,...
%              MultiScaleSupTable, verticalParaInd, groundParaInd, MultiScaleFlag, Sup2Para);
%[CoPM1 CoPM2 WeiCoP Sup Sup2ParaNew FixPara NewNuSup VertStickPointInd HoriStickPointInd] ...
%            = FixParaFreeCorner( Sup, Ray, HoriStickPointInd, VertStickPointInd,...
%              SepPointMeasureHori, SepPointMeasureVert, OccluList, 1); % may change 1 to logistic later

%[CoPM1 CoPM2 WeiCoP Sup Sup2ParaNew FixPara NewNuSup VertStickPointInd HoriStickPointInd] ...
%            = FixParaFreeCornerNew( Sup, Ray,  verticalParaInd, groundParaInd, NuSup, NuSupSize,OccluList, 1); % may change 1 to logistic later

  indexVertical = NewSup2Para(NuSup(verticalParaInd));
  indexGround = NewSup2Para(NuSup(groundParaInd));
%  indexGroundZ = Sup2ParaNew(NuSup(groundParaInd));
  ZeroMaskVertical = indexVertical ==0;
  ZeroMaskGround = indexGround ==0;
%  ZeroMaskGroundZ = indexGroundZ ==0;
  indexVertical = indexVertical(~ZeroMaskVertical);
  indexGround = indexGround(~ZeroMaskGround);
%  indexGroundZ = indexGroundZ(~ZeroMaskGroundZ);
  indexVertical = indexVertical*3-1;
  indexGroundX = indexGround*3-2;
  indexGroundZ = indexGround*3;


RayAllM = sparse(0,0);
%PosiM = sparse(0,0);
RayM = sparse(0,0);
DepthIveEst = [];
YPointer = [];
for i = NewNuSup
    mask = NewSup ==i;
    RayAllM = blkdiag( RayAllM, Ray(:,mask)');
    [yt x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(yt));
    YPointer = [YPointer; CenterY >= ceiling];
    if ~any( FixPara ==i )
       mask = logical(zeros(size(Sup)));
    end
       RayM = blkdiag( RayM, Ray(:,mask)');
       DepthIveEst = [DepthIveEst; 1./FitDepth(mask)];
end
PosiM = spdiags(1./DepthIveEst,0,size(DepthIveEst,1),size(DepthIveEst,1))*RayM;

NewNuSupSize = size(NewNuSup,2);
opt = sdpsettings('solver','sedumi');
ParaCorner = sdpvar(3*NewNuSupSize,1);
F = set(ParaCorner(3*(1:NewNuSupSize)-1).*YPointer<=0) + set(RayAllM*ParaCorner >=1/FarestDist)...
    +set(RayAllM*ParaCorner <=1/ClosestDist)...
    +set(ParaCorner([(NewSup2Para(FixPara)*3-2); (NewSup2Para(FixPara)*3-1); (NewSup2Para(FixPara)*3)]) == ...
    Para([(NewSup2Para(FixPara)*3-2); (NewSup2Para(FixPara)*3-1); (NewSup2Para(FixPara)*3)]));;
%    +set(ParaCorner(indexVertical) == 0);
%    +set(ParaCorner(indexGroundX) == 0);
%    +set(ParaCorner(indexGroundZ) == 0)...
%    +set(ParaCorner([(Sup2ParaNew(FixPara)*3-2); (Sup2ParaNew(FixPara)*3-1); (Sup2ParaNew(FixPara)*3)]) == ...
%    Para([(Sup2Para(FixPara)*3-2); (Sup2Para(FixPara)*3-1); (Sup2Para(FixPara)*3)]));
solvesdp(F,Center*norm( ((CoPM1 - CoPM2)*ParaCorner), 1 )...
          +1000*norm( ParaCorner(indexGroundX)./ normPara( groundParaInd(~ZeroMaskGround) )', 1)  ...
          +1000*norm( ParaCorner(indexGroundZ)./ normPara( groundParaInd(~ZeroMaskGround) )', 1) ...
          , opt);
          %+norm(((HoriStickM_i-HoriStickM_j)*ParaCorner).*EstDepHoriStick ,1)...
          %+norm(((VertStickM_i-VertStickM_j)*ParaCorner).*EstDepVertStick ,1)...
% CoPEstDepth
% PosiM*ParaCorner - 1,1)
% norm(RayM*ParaCorner - DepthIveEst,1)

%pause;
ParaCorner = double(ParaCorner);
ParaCorner = reshape(ParaCorner,3,[]);
FitDepthCorner = 100*ones(1,VertYNuDepth*HoriXNuDepth);
FitDepthCorner(NewSup~=0) = (1./sum(ParaCorner(:,NewSup2Para(NewSup(NewSup~=0))).*Ray(:,NewSup~=0),1))';
FitDepthCorner = reshape(FitDepthCorner,VertYNuDepth,[]);
[Position3DFitedCorner] = im_cr2w_cr(FitDepthCorner,permute(Ray,[2 3 1]));
[VrmlName] = vrml_test_faceset_goodSkyBoundary(	filename{k}, Position3DFitedCorner, FitDepthCorner, permute(Ray,[2 3 1]), ['Corner'], ...
[], [], 0, maskSkyEroded, maskG, 1, 0, a_default, b_default, Ox_default, Oy_default);
system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
delete([ScratchDataFolder '/vrml/' VrmlName]);
end
%return;
return;
