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
function  ReportPlaneParaMRF_Decompv2(Default, step, DepthFolder,...
          Sup,MedSup,depthMap,VarMap,RayOri, Ray,MedRay,maskSky,maskG,Algo,k, CornerList, OccluList,...
          MultiScaleSupTable, StraightLineTable, HBrokeBook, VBrokeBook,previoslyStored,...
          baseline);
% This function runs the RMF over the plane parameter of each superpixels

if nargin <20
   baseline = 0;
end

% initialize parameters
NOYALMIP = 1;
Dual = false;
displayFlag = false;
RenderVrmlFlag = true;
StickHori = 5;%0.1; % sticking power in horizontal direction
StickVert = 10;     % sticking power in vertical direction
Center = 2; % Co-Planar weight at the Center of each superpixel
HoriConf = 1; % set the confidant of the learned depth at the middle in Horizontal direction of the image
VertConf = 0.01; % set the confidant of the learned depth at the top of the image
BandWith = 1; % Nov29 1 Nov30 0.1 check result change to 0.1 12/1 0.1 lost detail
mapVert = linspace(VertConf,1,Default.VertYNuDepth); % modeling the gravity prior
mapHori = [linspace(HoriConf,1,round(Default.HoriXNuDepth/2)) fliplr(linspace(HoriConf,1,Default.HoriXNuDepth-round(Default.HoriXNuDepth/2)))];

% ========set the range of depth that our model in
ClosestDist = 1;
% set the FarestDist to very 5 times to median depth
FarestDist = 1.5*median(depthMap(:)); % tried on university % nogood effect but keep it since usually it the rangelike this   % change to 1.5 for church
% ================================================

ceiling = 0*Default.VertYNuDepth; % set the position of the ceiling, related to No plane coming back constrain % changed for newchurch
Name{1} = 'FraWOPri';
Name{2} = 'FraCoP';
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
ConerImprove = false;
FractionalDepthError = true;


% get rid of the error point and sky point in the depthMap
% set every depth bigger than FarestDistmeter to FarestDistmeters 
%CleanedDepthMap = (depthMapif ~previoslyStored >80).*medfilt2(depthMap,[4 4])+(depthMap<=80).*depthMap;
CleanedDepthMap = depthMap;
%CleanedDepthMap(depthMap>FarestDist) = FarestDist; % don't clean the point >80 sometimes it occlusion
%disp('Nu of depthMap>FarestDist');
%sum(sum(depthMap>FarestDist))
CleanedDepthMap(depthMap>FarestDist) = NaN; % don't clean the point >80 sometimes it occlusion
Posi3D = im_cr2w_cr(CleanedDepthMap,permute(Ray,[2 3 1]));

if ~previoslyStored

   NewMap = [rand(max(Sup(:)),3); [0 0 0]];
   % Clean the Sup near sky
   maskSky = Sup == 0;
   maskSkyEroded = imerode(maskSky, strel('disk', 4) );
   SupEpand = ExpandSup2Sky(Sup,maskSkyEroded);
   NuPatch = Default.HoriXNuDepth*Default.VertYNuDepth-sum(maskSky(:));

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
 % ===========================================================================================================================================
groundThreshold = cos([ zeros(1, Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)+10) linspace(0,15,ceil(Default.VertYNuDepth/2)-10)]*pi/180);  
  %  v1 15 v2 20 too big v3 20 to ensure non misclassified as ground.
%  verticalThreshold = cos(linspace(5,55,Default.VertYNuDepth)*pi/180); % give a vector of size 55 in top to down : 
  verticalThreshold = cos([ 5*ones(1,Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)) linspace(5,55,ceil(Default.VertYNuDepth/2))]*pi/180); % give a vector of size 55 in top to down : 
  % 50 means suface norm away from y axis more than 50 degree
 % ===========================================================================================================================================

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
YPosition = [];
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
    YPosition = [YPosition; CenterY];
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
      PosiM = blkdiag(PosiM,Posi3D(:,mask)');%.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      VarM = [VarM; VarMap(mask).*(mapVert(yt)').*( mapHori(x)')];
      RayMd = blkdiag(RayMd,RayOri(:,mask)'.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      DepthInverseMCent = [DepthInverseMCent; 1./median(CleanedDepthMap(mask)).*SupNuPatch(i)* mapVert(CenterY)'*mapHori(CenterX)];
      DepthInverseM = [DepthInverseM; 1./CleanedDepthMap(mask).* mapVert(yt)'.*mapHori(x)'];
    else
      Pa2CenterRatio = CleanedDepthMap(CenterY,CenterX)./CleanedDepthMap(mask);
      if sum(mask(:)) > 0
         RayMtilt = blkdiag(RayMtilt, ...
             ( Pa2CenterRatio(:,[1 1 1]).*repmat(RayOri(:,CenterY,CenterX)',[ SupNuPatch(i) 1])- RayOri(:,mask)'));
      else
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
ClosestNList = [ Sup(find(BounaryPHori==1)) Sup(find(BounaryPHori==1)+Default.VertYNuDepth);...
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
   NuSTList = size(CoPSTList,1);
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
    %NuNei-NuSTList
%    i
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

SupPixelNeighborList = sparse(NuSupSize, NuSupSize);
MAX_POINTS_STITCH = inf;    % the actual code will be modified in another copy of the file

for i = find(BounaryPHori==1)'
    j = i+Default.VertYNuDepth;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
    if SupPixelNeighborList(Sup(i),Sup(j)) > MAX_POINTS_STITCH      % I.e., increase weight, and do not add to the matrix
        SupPixelNeighborList(Sup(i),Sup(j)) = SupPixelNeighborList(Sup(i),Sup(j)) + 1;
        WeightHoriNeighborStitch = SupPixelNeighborList(Sup(i),Sup(j)) / MAX_POINTS_STITCH;     % weight will be distributed between MAX_POINTS_STITCH
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
depthfile = strrep(Default.filename{k},'img','depth_learned'); %


% Start Decompose the image align with superpixels ==================================================================================
% define the decomposition in X direction only

% ============== parameters for the decomposition problem
XNuDecompose = 1; 
YNuDecompose = 1;
% ============ parameters for the decomposition problem


 
TotalRectX = 2*XNuDecompose-1;
TotalRectY= 2*YNuDecompose-1;
PlanePara = NaN*ones(3*NuSupSize,1); % setup the lookuptable for the solved plane parameter

opt = sdpsettings('solver','sedumi','cachesolvers',1);
%     opt = sdpsettings('solver','lpsolve','cachesolvers',1);
%    opt = sdpsettings('solver','glpk','cachesolvers',1);    

for k = 0:(TotalRectX-1)
  l = rem(k*2,(TotalRectX));
  RangeX = (1+ceil(Default.HoriXNuDepth/XNuDecompose)*l/2):...
            min((1+ceil(Default.HoriXNuDepth/XNuDecompose)*(l/2+1)),Default.HoriXNuDepth);
  RangeX = ceil(RangeX);
  for q = 0:(TotalRectY-1)
    l = rem(q*2,(TotalRectY));
    RangeY = (1+ceil(Default.VertYNuDepth/YNuDecompose)*l/2):...
              min((1+ceil(Default.VertYNuDepth/YNuDecompose)*(l/2+1)),Default.VertYNuDepth);
    RangeY = ceil(RangeY);
    mask = zeros(size(Sup));
    mask(RangeY,RangeX) = 1;
    mask =logical(mask);
    SubSup = sort(setdiff(reshape( Sup(RangeY,RangeX),[],1),0))';
    BoundarySup = [];
    for m = SubSup
        maskList = ClosestNList(:,1) == m;
        BoundarySup = [ BoundarySup ClosestNList(maskList,2)'];
    end
    BoundarySup = unique(setdiff(BoundarySup,[0 SubSup] ));
    % chech if BoundarySup non-NaN in PlanePara
    checkNoNNaN = ~isnan(PlanePara(Sup2Para(BoundarySup)*3));
    BoundarySup = BoundarySup(checkNoNNaN);
    TotalSup = sort([SubSup BoundarySup]);

    TotalSupPtr = [ Sup2Para(TotalSup)*3-2;...
                    Sup2Para(TotalSup)*3-1;...
                    Sup2Para(TotalSup)*3];
    TotalSupPtr = TotalSupPtr(:);
    BoundarySupPtr = [ Sup2Para(BoundarySup)*3-2;...
                       Sup2Para(BoundarySup)*3-1;...
                       Sup2Para(BoundarySup)*3];
    BoundarySupPtr =BoundarySupPtr(:);
    NuSubSupSize = size(TotalSup,2);
    TotalSup2Para = sparse(1,max(TotalSup));
    TotalSup2Para(TotalSup) = 1:NuSubSupSize;
    BoundarySupPtrSub = [ TotalSup2Para(BoundarySup)*3-2;...
                          TotalSup2Para(BoundarySup)*3-1;...
                          TotalSup2Para(BoundarySup)*3];
    BoundarySupPtrSub =BoundarySupPtrSub(:);
    
    % clearn RayAllM PosiM CoPM1 HoriStickM_i VertStickM_i
    NewRayAllM = RayAllM(:,TotalSupPtr);
    tar = sum(NewRayAllM ~= 0,2) ==3;
    NewRayAllM = NewRayAllM(tar,:);
    NewPosiM = PosiM(:,TotalSupPtr);
    tar = sum(NewPosiM ~= 0,2) ==3;
    NewPosiM = NewPosiM(tar,:);
    NewVarM = VarM(tar);
    NewCoPM = CoPM1(:,TotalSupPtr) - CoPM2(:,TotalSupPtr);
    tar = sum(NewCoPM ~= 0,2) ==6;
    NewCoPM = NewCoPM(tar,:);
    NewCoPEstDepth = CoPEstDepth(tar);   
    NewHoriStickM = HoriStickM_i(:,TotalSupPtr)-HoriStickM_j(:,TotalSupPtr);
    tar = sum(NewHoriStickM ~= 0,2) ==6;
    NewHoriStickM = NewHoriStickM(tar,:);
    NewEstDepHoriStick = EstDepHoriStick(tar);
    NewVertStickM = VertStickM_i(:,TotalSupPtr)-VertStickM_j(:,TotalSupPtr);
    tar = sum(NewVertStickM ~= 0,2) ==6;
    NewVertStickM = NewVertStickM(tar,:);
    NewEstDepVertStick = EstDepVertStick(tar);
    
    for i = step
        ParaPPCP = sdpvar(3*NuSubSupSize,1);
        F = set(ParaPPCP(3*(1:NuSubSupSize)-1).*YPointer(Sup2Para(TotalSup))<=0)...
            +set(NewRayAllM*ParaPPCP <=1/ClosestDist)...
            +set(NewRayAllM*ParaPPCP >=1/FarestDist)...
            +set(ParaPPCP(BoundarySupPtrSub) == PlanePara(BoundarySupPtr) ); % Special constrain for decomp fix the solved neighbor plane parameter       
%         +set(RayAllOriM(:,TotalSupPtr)*ParaPPCP >=1/FarestDist)...
%         +set(RayAllOriM(:,TotalSupPtr)*ParaPPCP <=1/ClosestDist);
% First fit the plane to find the estimated plane parameters
% If PosiM contain NaN data the corresponding Plane Parameter will also be NaN
    if i == 1 % W/O  priors
%    solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./(abs(VarM)),1)...
       solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./exp(abs(VarM)/BandWith),1)...
             , opt);
    elseif i ==2 % Coner
       solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./exp(abs(VarM)/BandWith),1)...
             +Center*norm(((CoPM1 - CoPM2)*ParaPPCP).*CoPEstDepth, 1)...
             , opt);
    else % all the priors
%    sum(sum(isnan(PosiM)))
       disp('Whole MRF')
%     solvesdp(F,norm( (PosiM*ParaPPCP-ones(size(PosiM,1),1))./exp(abs(VarM)/BandWith),1)...
%              +Center*norm(((CoPM1 - CoPM2)*ParaPPCP).*CoPEstDepth, 1)...
%              +norm(((HoriStickM_i-HoriStickM_j)*ParaPPCP).*EstDepHoriStick,1)...
%              +norm(((VertStickM_i-VertStickM_j)*ParaPPCP).*EstDepVertStick,1)...
%              , opt);
   

% =========  Debugging the code for numerical accuracy  ==========



% ================================================================

 
       if NOYALMIP
          sol = solvesdp(F, norm( (NewPosiM*ParaPPCP-ones(size(NewPosiM,1),1)) ./ exp(abs(NewVarM)/BandWith),1)...
              		+ Center*norm(((NewCoPM)*ParaPPCP).*NewCoPEstDepth, 1)...
              		+ norm(((NewHoriStickM)*ParaPPCP).*NewEstDepHoriStick,1)...
              		+ norm(((NewVertStickM)*ParaPPCP).*NewEstDepVertStick,1), ...
			opt);
       else

       % ============Using Sedumi directly ==================================
          if Dual 
          % trial for dual form


          else
            % trial for MS_E212 method: Primal form
            %  D1 = PosiM;
            %  D2 = (CoPM1 - CoPM2);
            %  D3 = (HoriStickM_i-HoriStickM_j);
            %  D4 = (VertStickM_i-VertStickM_j);
            %  b = -ones(size(PosiM,1),1);
  
            %  C1 = RayAllM;
            %  C2 = -RayAllM
            %  clear tempSparse;
            %  e1 = 1/ClosestDist*ones(size(RayAllM,1),1);
            %  e2 = -1/ FarestDist*ones(size(RayAllM,1),1);


            %  e2 = sparse(3*NuSupSize,1);
            %  C2 = spdiags(tempSparse,0,3*NuSupSize,3*NuSupSize);

            %  t1 = 1./exp(abs(VarM)/BandWith);
            %  t2 = CoPEstDepth*Center;
            %  t3 = EstDepHoriStick;
            %  t4 = EstDepVertStick;

            sizeX = size(PosiM,2);
            sizeD1 = size(PosiM,1);
            sizeD2 = size(CoPM1,1);
            sizeD3 = size(HoriStickM_i,1);
            sizeD4 = size(VertStickM_i,1);
            sizeC1 = size(RayAllM,1);
            sizeC2 = sizeC1;
            %   sizeC2 = size(spdiags(tempSparse,0,3*NuSupSize,3*NuSupSize),1);
  
  
            A = [ -2*PosiM ... % x
                  -speye(sizeD1,sizeD1) ... % V_1
                  sparse(sizeD1,(sizeD2+sizeD3+sizeD4) );...
                  ...
                  -2*(CoPM1 - CoPM2) ... % x
                  sparse(sizeD2,sizeD1) ... 
                  -spdiags(ones(sizeD2,1),0,sizeD2,sizeD2) ... % V_2
                  sparse(sizeD2, (sizeD3+sizeD4));...
                  ...
                  -2*(HoriStickM_i-HoriStickM_j) ... %x
                  sparse(sizeD3, (sizeD1+sizeD2) ) ...
                  -spdiags(ones(sizeD3,1),0,sizeD3,sizeD3) ... % V_3
                  sparse(sizeD3,sizeD4 );...
                  ...
                  -2*(VertStickM_i-VertStickM_j) ... %x
                  sparse(sizeD4, (sizeD1+sizeD2+sizeD3)) ...
                  -spdiags(ones(sizeD4,1),0,sizeD4,sizeD4); ... % V_4
                  ...
                  RayAllM ...
                  sparse(sizeC1,(sizeD1+sizeD2+sizeD3+sizeD4) ); ...
                  ...
                  -RayAllM ...
                  sparse(sizeC2,(sizeD1+sizeD2+sizeD3+sizeD4) ); ...
            ];  

            b = [ -2*ones(size(PosiM,1),1); ...
                  sparse(sizeD2+sizeD3+sizeD4,1);...
                  1/ClosestDist*ones(size(RayAllM,1),1);...
                  -1/FarestDist*ones(size(RayAllM,1),1);...
                ];
            H = [ (1./exp(abs(VarM)/BandWith))'*(PosiM)+...
                  (CoPEstDepth*Center)'*(CoPM1 - CoPM2)+...
                  (EstDepHoriStick)'*((HoriStickM_i-HoriStickM_j))+...
                  (EstDepVertStick)'*((VertStickM_i-VertStickM_j)) ...
                  ...
                  (1./exp(abs(VarM)/BandWith))' ...
                  ...
                  CoPEstDepth'*Center ...
                  ...
                  EstDepHoriStick' ...
                  ...
                  EstDepVertStick' ...
                ];
            lb = [-Inf*ones(sizeX,1); ...
                  sparse((sizeD1+sizeD2+sizeD3+sizeD4),1)];
            tempSparse = sparse(3*NuSupSize,1);
            tempSparse(3*(1:NuSupSize)-1) = YPointer;
            ub = Inf*ones(sizeX,1);
            ub(logical(tempSparse)) = 0;
            ub = [ub; Inf*ones((sizeD1+sizeD2+sizeD3+sizeD4),1)];
  
            %   options = optimset('LargeScale','on');
            %  K.f = sizeX;
            %  ParaPPCP = J*sedumi(A,b,H,K);
            %   linOut = linprog(H',A,b,[],[],lb,ub,[],options);
            [obj,x,duals,stat] = lp_solve(-H',A,b,-1*ones(size(b)),lb,ub);
             ParaPPCP = x(1:sizeX);
            %   opt = sdpsettings('solver','sedumi');
            %   ParaPPCP = sdpvar(sizeX + sizeD1 +sizeD2+sizeD3+sizeD4);
            %   F = set(A*ParaPPCP <= b) + set(ParaPPCP>=lb);
            %   ParaPPCP = J*solvesdp(F,H*ParaPPCP,opt);
          end
       end
    end

    ParaPPCP = double(ParaPPCP);
%     sum(isnan(ParaPPCP))
    yalmip('clear');
    PlanePara(TotalSupPtr) = ParaPPCP;

% %    SepPointMeasureHori = 1./(HoriStickM_i*ParaPPCP)-1./(HoriStickM_j*ParaPPCP);
% %    SepPointMeasureVert = 1./(VertStickM_i*ParaPPCP)-1./(VertStickM_j*ParaPPCP);
%     ParaPPCP = reshape(ParaPPCP,3,[]);
%     %any(any(isnan(ParaPPCP)))
%     % porject the ray on planes to generate the ProjDepth
%     FitDepthPPCP = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
%     FitDepthPPCP(~maskSkyEroded & mask) = (1./sum(ParaPPCP(:,TotalSup2Para(SupEpand(~maskSkyEroded & mask))).*Ray(:,~maskSkyEroded & mask),1))';
%     FitDepthPPCP = reshape(FitDepthPPCP,Default.VertYNuDepth,[]);
%     % storage for comparison purpose ======================
% %    depthMap = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
% %    depthMap(~maskSkyEroded | mask) = (1./sum(ParaPPCP(:,TotalSup2Para(SupEpand(~maskSkyEroded | mask))).*RayOri(:,~maskSkyEroded | mask),1))';
% %    depthMap = reshape(depthMap,Default.VertYNuDepth,[]);
% %    if baseline == 0
% %       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '/']);
% %       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '/' depthfile '.mat'],'depthMap');
% %    elseif baseline == 1
% %       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/']);
% %       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/' depthfile '.mat'],'depthMap');
% %    else
% %       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/']);
% %       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/' depthfile '.mat'],'depthMap');
% %    end
%   
%     % =====================================================
%     [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
%     if true %RenderVrmlFlag && i==3
%          Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
%          Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
%          RR =permute(Ray,[2 3 1]);
%          temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
%          PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),[Default.VertYNuDepth Default.HoriXNuDepth 1])+repmat(cat(3,Default.Ox_default,Default.Oy_default),[Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
%          PositionTex = permute(PositionTex,[2 3 1]);
%          WrlFacest(Position3DFitedPPCP,PositionTex,Sup,Default.filename{1},'./')
% 
% %        [VrmlName] = vrml_test_faceset_goodSkyBoundary(	Default.filename{k}, Position3DFitedPPCP, FitDepthPPCP, permute(Ray,[2 3 1]), [Name{i} DepthFolder 'ExpVar'], ...
% %                     [], [], 0, maskSkyEroded, maskG, 1, 0, Default.a_default, Default.b_default, Default.Ox_default, Default.Oy_default);
% %        system(['gzip -9 -c ' Default.ScratchDataFolder '/vrml/' VrmlName ' > ' Default.ScratchDataFolder '/vrml/' VrmlName '.gz']);
%        %delete([Default.ScratchDataFolder '/vrml/' VrmlName]);
%     end
%    save([Default.ScratchDataFolder '/data/PreDepth/FitDepthPPCP' num2str(k) '.mat'],'FitDepthPPCP','SepPointMeasureHori',...
%      'SepPointMeasureVert','VertStickPointInd','HoriStickPointInd');
%save([Default.ScratchDataFolder '/data/All.mat']);
  end
  end
end
% build the whole image
    PlanePara = reshape(PlanePara,3,[]);
    %any(any(isnan(PlanePara)))
    % porject the ray on planes to generate the ProjDepth
    FitDepthPPCP = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
    FitDepthPPCP(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded ))).*Ray(:,~maskSkyEroded ),1))';
    FitDepthPPCP = reshape(FitDepthPPCP,Default.VertYNuDepth,[]);
    % storage for comparison purpose ======================
%    depthMap = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
%    depthMap(~maskSkyEroded | mask) = (1./sum(PlanePara(:,TotalSup2Para(SupEpand(~maskSkyEroded | mask))).*RayOri(:,~maskSkyEroded | mask),1))';
%    depthMap = reshape(depthMap,Default.VertYNuDepth,[]);
%    if baseline == 0
%       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '/']);



%       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '/' depthfile '.mat'],'depthMap');
%    elseif baseline == 1
%       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/']);
%       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline/' depthfile '.mat'],'depthMap');
%    else
%       system(['mkdir ' Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/']);
%       save([Default.ScratchDataFolder '/' Name{i} '_' DepthFolder '_baseline2/' depthfile '.mat'],'depthMap');
%    end

    % =====================================================
    [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
    if false %RenderVrmlFlag && i==3
         Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
         Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
         RR =permute(Ray,[2 3 1]);
         temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
         PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),[Default.VertYNuDepth Default.HoriXNuDepth 1])+repmat(cat(3,Default.Ox_default,Default.Oy_default),[Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
         PositionTex = permute(PositionTex,[2 3 1]);
         WrlFacestHroiReduce(Position3DFitedPPCP,PositionTex,Sup,Default.filename{1},'./')
    end

%return;
%==================Finished for one step MRF==========================================================================================================

NoSecondStep = 1;
if NoSecondStep
	return;
end

% ====================================following are 2nd step MRF to give more visually pleasing result=======================================
;% generating new PosiMPPCP using the new position

% save([Default.ScratchDataFolder '/data/VertShiftVrml.mat' ] );
%  groundThreshold = cos(5*pi/180);  % make small range
%  verticalThreshold = cos(50*pi/180);
  normPara = norms(PlanePara);
  normalizedPara = PlanePara ./ repmat( normPara, [3 1]);
  groundPara = abs(normalizedPara(2,:)) >= groundThreshold(YPosition);
  groundParaInd = find(groundPara);
  verticalPara = abs(normalizedPara(2,:)) <= verticalThreshold(YPosition); % change to have different range of vertical thre ================
  verticalParaInd = find(verticalPara);

  indexVertical = find( verticalPara)*3-1;
  indexGroundX = find( groundPara)*3-2;
  indexGroundZ = find( groundPara)*3;

  PosiMPPCP = sparse(0,0);
  VarM2 = sparse(0,0);
  %  VertVar = 10^(-2);
  % forming new supporting matrix using new depth and get rid of the support of the vertical plane
  for i = NuSup
      mask = Sup == i;
      if any(verticalParaInd == Sup2Para(i))
         mask = logical(zeros(size(Sup)));
      end
         PosiMPPCP = blkdiag(PosiMPPCP, Posi3D(:,mask)');
         VarM2 = [VarM2; VarMap(mask)];         
  end

% Start Decompose image =========================
  XNuDecompose = 2; 
  YNuDecompose = 1; 
  TotalRectX = 2*XNuDecompose-1;
  TotalRectY = 2*YNuDecompose-1;
  PlanePara = NaN*ones(3*NuSupSize,1); % setup the lookuptable for the solved plane parameter

  opt = sdpsettings('solver','sedumi','cachesolvers',1);
  %     opt = sdpsettings('solver','lpsolve','cachesolvers',1);
  %    opt = sdpsettings('solver','glpk','cachesolvers',1);    

  for k = 0:(TotalRectX-1)
    l = rem(k*2,(TotalRectX));
    RangeX = (1+ceil(Default.HoriXNuDepth/XNuDecompose)*l/2):...
              min((1+ceil(Default.HoriXNuDepth/XNuDecompose)*(l/2+1)),Default.HoriXNuDepth);
    RangeX = ceil(RangeX);
    for q = 0:(TotalRectY-1)
      l = rem(q*2,(TotalRectY));
      RangeY = (1+ceil(Default.VertYNuDepth/YNuDecompose)*l/2):...
                min((1+ceil(Default.VertYNuDepth/YNuDecompose)*(l/2+1)),Default.VertYNuDepth);
      RangeY = ceil(RangeY);
      mask = zeros(size(Sup));
      mask(RangeY,RangeX) = 1;
      mask =logical(mask);
      SubSup = sort(setdiff(reshape( Sup(RangeY,RangeX),[],1),0))';
      BoundarySup = [];
      for m = SubSup
          maskList = ClosestNList(:,1) == m;
          BoundarySup = [ BoundarySup ClosestNList(maskList,2)'];
      end
      BoundarySup = unique(setdiff(BoundarySup,[0 SubSup] ));

      % chech if BoundarySup non-NaN in PlanePara
      checkNoNNaN = ~isnan(PlanePara(Sup2Para(BoundarySup)*3));
      BoundarySup = BoundarySup(checkNoNNaN);
      TotalSup = sort([SubSup BoundarySup]);

      TotalSupPtr = [ Sup2Para(TotalSup)*3-2;...
                      Sup2Para(TotalSup)*3-1;...
                      Sup2Para(TotalSup)*3];
      TotalSupPtr = TotalSupPtr(:);
      SubSupPtr = [ Sup2Para(SubSup)*3-2;...
                    Sup2Para(SubSup)*3-1;...
                    Sup2Para(SubSup)*3];
      SubSupPtr = SubSupPtr(:);
      BoundarySupPtr = [ Sup2Para(BoundarySup)*3-2;...
                         Sup2Para(BoundarySup)*3-1;...
                         Sup2Para(BoundarySup)*3];
      BoundarySupPtr =BoundarySupPtr(:);
      NuSubSupSize = size(TotalSup,2);
      TotalSup2Para = sparse(1,max(TotalSup));
      TotalSup2Para(TotalSup) = 1:NuSubSupSize;
      BoundarySupPtrSub = [ TotalSup2Para(BoundarySup)*3-2;...
                            TotalSup2Para(BoundarySup)*3-1;...
                            TotalSup2Para(BoundarySup)*3];
      BoundarySupPtrSub =BoundarySupPtrSub(:);
      SubSupPtrSub = [ TotalSup2Para(SubSup)*3-2;...
                       TotalSup2Para(SubSup)*3-1;...
                       TotalSup2Para(SubSup)*3];
      SubSupPtrSub =SubSupPtrSub(:);
    
      % clearn RayAllM PosiM CoPM1 HoriStickM_i VertStickM_i
      NewRayAllM = RayAllM(:,TotalSupPtr);
      tar = sum(NewRayAllM ~= 0,2) ==3;
      NewRayAllM = NewRayAllM(tar,:);
      NewPosiMPPCP = PosiMPPCP(:,TotalSupPtr);
      tar = sum(NewPosiMPPCP ~= 0,2) ==3;
      NewPosiMPPCP = NewPosiMPPCP(tar,:);
      NewVarM = VarM2(tar);
      NewCoPM = CoPM1(:,TotalSupPtr) - CoPM2(:,TotalSupPtr);
      tar = sum(NewCoPM ~= 0,2) ==6;
      NewCoPM = NewCoPM(tar,:);
      NewCoPEstDepth = CoPEstDepth(tar);   
      NewHoriStickM = HoriStickM_i(:,TotalSupPtr)-HoriStickM_j(:,TotalSupPtr);
      tar = sum(NewHoriStickM ~= 0,2) ==6;
      NewHoriStickM = NewHoriStickM(tar,:);
      NewEstDepHoriStick = EstDepHoriStick(tar);
      NewVertStickM = VertStickM_i(:,TotalSupPtr)-VertStickM_j(:,TotalSupPtr);
      tar = sum(NewVertStickM ~= 0,2) ==6;
      NewVertStickM = NewVertStickM(tar,:);
      NewEstDepVertStick = EstDepVertStick(tar);


      Para = sdpvar(3*NuSubSupSize,1);
      F = set(Para(3*(1:NuSubSupSize)-1).*YPointer(Sup2Para(TotalSup))<=0)...
          +set(NewRayAllM*Para <=1/ClosestDist)...
          +set(NewRayAllM*Para >=1/FarestDist)...
          +set(Para(BoundarySupPtrSub) == PlanePara(BoundarySupPtr) )...
          +set( Para( TotalSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))*3-1) == 0); 
          % Special constrain for decomp fix the solved neighbor plane parameter       

          %      +set(Para(indexGroundX) == 0)...
          %      +set(Para(indexGroundZ) == 0);

      if FractionalDepthError
%         size(PosiMPPCP)
         solvesdp(F,norm( ( NewPosiMPPCP*Para-ones(size(NewPosiMPPCP,1),1))./exp(abs(NewVarM)/BandWith),1)...
                 +Center*norm(( NewCoPM*Para).*NewCoPEstDepth, 1)...
                 +norm(( NewHoriStickM*Para).*NewEstDepHoriStick,1)...
                 +norm(( NewVertStickM*Para).*NewEstDepVertStick,1)...
                 +10*norm( ( Para( TotalSup2Para( NuSup( (intersect( indexGroundX,SubSupPtr ) +2)/3))*3-2))./...
                            normPara( intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) )', 1)...
                 +10*norm( ( Para( TotalSup2Para( NuSup( (intersect( indexGroundZ,SubSupPtr ) )/3))*3))./...
                            normPara( intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) )', 1) ...
                 , opt);
%             sqrt( max(CoPM1*ParaPPCP(:), 1/FarestDist).*max(CoPM2*ParaPPCP(:), 1/FarestDist)), 1)...
%             +norm(((VertStickM_i-VertStickM_j)*Para)./...
%             sqrt( max(VertStickM_i*ParaPPCP(:), 1/FarestDist).*max(VertStickM_j*ParaPPCP(:), 1/FarestDist)),1)...
%   pause;
    %         +1000*norm( Para(indexVertical)./ normPara( find(verticalPara) )',1) ...
%            +norm(((HoriStickM_i-HoriStickM_j)*Para)./sqrt((VertStickM_i*ParaPPCP(:)).*(VertStickM_j*ParaPPCP(:))),1)...
      else % not used anymore
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

     Para = double(Para);
     %sum(isnan(Para))
     yalmip('clear');
     PlanePara(TotalSupPtr) = Para;
    end
  end

  %pause
  PlanePara = reshape(PlanePara,3,[]);
  FitDepth = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
  FitDepth(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
  FitDepth = reshape(FitDepth,Default.VertYNuDepth,[]);
  %sum(isnan(FitDepth(:)))
  [Position3DFited] = im_cr2w_cr(FitDepth,permute(Ray,[2 3 1]));
  Position3DFited(3,:) = -Position3DFited(3,:);
  Position3DFited = permute(Position3DFited,[2 3 1]);
  RR =permute(Ray,[2 3 1]);
  temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
  PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),...
                [Default.VertYNuDepth Default.HoriXNuDepth 1])...
                +repmat(cat(3,Default.Ox_default,Default.Oy_default),...
                [Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
  PositionTex = permute(PositionTex,[2 3 1]);
  WrlFacestHroiReduce(Position3DFited,PositionTex,Sup,Default.filename{1},'./')

% save depth Map ++++++++++++++++
%       depthMap = FitDepth;
%       system(['mkdir ' Default.ScratchDataFolder '/VNonSupport_' DepthFolder '/']);
%       save([Default.ScratchDataFolder '/VNonSupport_' DepthFolder '/' depthfile '.mat'],'depthMap');
% =============================
% return;
