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
function [SupMerg MedSupMerg] = FitPlaneLaser_CoPlane(...
          Sup,MedSup,depthMap,Ray,MedRay,maskSky,maskG,Algo,k, CornerList, OccluList,previoslyStored);
% This function runs the RMF for the plane parameter of each superpixels
% output:
% newSup : newSuperpixel index 
% PlanePara : Plane Parameter corresponding to the new Superpuixel index

displayFlag = false;

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

TrialName ='LaserL1PosiStickAlign'
Mult =1;
vari =0.1;
shift = 10;
StickHori = 1;
StickVert = 10;
HoriConf = 0;
VertConf = 0;
mapVert = logspace(VertConf,0,VertYNuDepth) % modeling the gravity prior
mapHori = [logspace(HoriConf,0,round(HoriXNuDepth/2)) fliplr(logspace(HoriConf,0,HoriXNuDepth-round(HoriXNuDepth/2)))] % modeling the gravity prior
gravity =true;
threhConer = 0.5;
threhJump = 1.2;
ceiling = 0*VertYNuDepth;
% get rid of the error point and sky point in the depthMap
% set every depth bigger than 80meter as error point and sky point
%CleanedDepthMap = (depthMapif ~previoslyStored >80).*medfilt2(depthMap,[4 4])+(depthMap<=80).*depthMap;
CleanedDepthMap = depthMap;
%if ~learned
%   CleanedDepthMap(depthMap>80) = NaN; % don't clean the point >80 sometimes it occlusion
%end
Posi3D = im_cr2w_cr(CleanedDepthMap,permute(Ray,[2 3 1]));

%previoslyStored = true;
if ~previoslyStored

NewMap = [rand(max(Sup(:)),3); [0 0 0]];
% Clean the Sup
%maskSky = imdilate(maskSky, strel('disk', 3) );
%[Sup,MedSup]=CleanedSup(Sup,MedSup,maskSky);
maskSky = Sup == 0;
maskSkyEroded = imerode(maskSky, strel('disk', 4) );
SupEpand = ExpandSup2Sky(Sup,maskSkyEroded);
NuPatch = HoriXNuDepth*VertYNuDepth-sum(maskSky(:));


% find out the neighbors
%[nList]=GenSupNeighbor(Sup,maskSky); % Cell array : [ Supindex, Neighbor List of SupIndex] 
NuSup = setdiff(unique(Sup)',0);
NuSup = sort(NuSup);
NuSupSize = size(NuSup,2);

% Sup index and planeParameter index inverse map
Sup2Para = sparse(1,max(Sup(:)));
Sup2Para(NuSup) = 1:NuSupSize;

% Generate the Matrix for MRF
tic
PosiM = sparse(0,0);
RayMd = sparse(0,0);
RayAllM = sparse(0,0);
RayMtilt = sparse(0,0);
RayMCent = sparse(0,0);
DepthInverseMCent = [];
DepthInverseM = [];
YPointer = [];
beta = [];
EmptyIndex = [];
for i = NuSup
    mask = Sup ==i;
    RayAllM = blkdiag( RayAllM, Ray(:,mask)');
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
             ( Pa2CenterRatio(:,[1 1 1]).*repmat(Ray(:,CenterY,CenterX)',[ SupNuPatch(i) 1])- Ray(:,mask)'));
      else
         RayMtilt = blkdiag(RayMtilt, Ray(:,mask)');
      end
      RayMCent = blkdiag(RayMCent, Ray(:,CenterY,CenterX)'*SupNuPatch(i)*mapVert(CenterY)*mapHori(CenterX));
      PosiM = blkdiag(PosiM,Posi3D(:,mask)');
      RayMd = blkdiag(RayMd,Ray(:,mask)'.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      DepthInverseMCent = [DepthInverseMCent; 1./median(CleanedDepthMap(mask)).*SupNuPatch(i)* mapVert(CenterY)'*mapHori(CenterX)];
      DepthInverseM = [DepthInverseM; 1./CleanedDepthMap(mask).* mapVert(yt)'.*mapHori(x)'];
    else
      Pa2CenterRatio = CleanedDepthMap(CenterY,CenterX)./CleanedDepthMap(mask);
      if sum(mask(:)) > 0
         RayMtilt = blkdiag(RayMtilt, ...
             ( Pa2CenterRatio(:,[1 1 1]).*repmat(Ray(:,CenterY,CenterX)',[ SupNuPatch(i) 1])- Ray(:,mask)'));
      else
         RayMtilt = blkdiag(RayMtilt, Ray(:,mask)');
      end
      RayMCent = blkdiag(RayMCent, Ray(:,CenterY,CenterX)'*SupNuPatch(i));
      PosiM = blkdiag(PosiM,Posi3D(:,mask)');
      RayMd = blkdiag(RayMd,Ray(:,mask)');
      DepthInverseMCent = [DepthInverseMCent; 1./median(CleanedDepthMap(mask)).*SupNuPatch(i)];
      DepthInverseM = [DepthInverseM; 1./CleanedDepthMap(mask)];
    end
  else
     RayMtilt = blkdiag(RayMtilt, Ray(:,mask)');
     RayMCent = blkdiag(RayMCent, Ray(:,mask)');
     PosiM = blkdiag(PosiM, Posi3D(:,mask)');
     RayMd = blkdiag(RayMd, Ray(:,mask)');
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
NuNei = size(ClosestNList,1);
CoPM1 = sparse(0,3*NuSupSize);
CoPM2 = sparse(0,3*NuSupSize);
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
%    [yt x] = find(mask);
%    CenterX = round(median(x));
%    CenterY = round(median(yt));
    
    temp1 = sparse(1, 3*NuSupSize);
    temp2 = sparse(1, 3*NuSupSize);
    temp1(:,(Sup2Para(ClosestNList(i,1))*3-2): Sup2Para(ClosestNList(i,1))*3) = Ray(:,CenterY,CenterX)';
    temp2(:,(Sup2Para(ClosestNList(i,2))*3-2): Sup2Para(ClosestNList(i,2))*3) = Ray(:,CenterY,CenterX)';
    CoPM1 = [CoPM1; temp1];
    CoPM2 = [CoPM2; temp2];
    tempWeiCoP = [SizeMaskAll];
    
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
    CoPM1 = [CoPM1; temp1];
    CoPM2 = [CoPM2; temp2];
    tempWeiCoP = [tempWeiCoP; SizeMaskAll];
    WeiCoP = [WeiCoP; tempWeiCoP];
%  end
end%=========================================================================================================

% find the boundary point that might need to be stick ot each other==========================================
HoriStickM = sparse(1,2+3*NuSupSize);
for i = find(BounaryPHori==1)'
    j = i+VertYNuDepth;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
%  if ~OccluList(sum( ClosestNList == repmat(sort([Sup(i) Sup(j)]), [NuNei  1]),2) == 2)
    Target(1) = Sup2Para(Sup(i));
    Target(2) = Sup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
    betaTemp = StickHori;%*(DistStickLengthNormWei.^2)*beta(Target(I));
    temp = sparse(3,NuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    temp(:,Target(2)) = -rayBoundary(:,2);
    HoriStickM = [HoriStickM; [sort([Sup(i) Sup(j)]) betaTemp*temp(:)']];
%  end
end
VertStickM = sparse(1,2+3*NuSupSize);
for i = find(BounaryPVert==1)'
    j = i+1;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
%  if ~OccluList(sum( ClosestNList == repmat(sort([Sup(i) Sup(j)]), [NuNei  1]),2) == 2)
    Target(1) = Sup2Para(Sup(i));
    Target(2) = Sup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
    betaTemp = StickVert;%*(DistStickLengthNormWei.^2)*beta(Target(I));
    temp = sparse(3,NuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    temp(:,Target(2)) = -rayBoundary(:,2);
    VertStickM = [VertStickM; [sort([Sup(i) Sup(j)]) betaTemp*temp(:)']];
%  end
end

% 2nd: Associate Plane Parameter term and Interactive Co-Planar term
opt = sdpsettings('solver','sedumi');
ParaPPCP = sdpvar(3*NuSupSize,1);
F = set(RayAllM*ParaPPCP >=0.01);
%F = set(ParaPPCP(3*(1:NuSupSize)-1)<=0) + set(RayAllM*ParaPPCP >=0.01);
% First fit the plane to find the estimated plane parameters
% If PosiM contain NaN data the corresponding Plane Parameter will also be NaN
%solvesdp([], norm( spdiags(ConfRayMean',0,size(RayMeanM,1),size(RayMeanM,1))*(RayMeanM*ParaOri(:) - RayMeanM*ParaPPCP),1)...
%solvesdp([],norm( PosiM*ParaPPCP-ones(size(PosiM,1),1),1)...
%solvesdp(F,norm( RayMCent*ParaPPCP - DepthInverseMCent,1) ...
solvesdp(F,norm( RayMd*ParaPPCP - DepthInverseM,1)... 
         +norm((CoPM1 - CoPM2)*ParaPPCP,1), opt);
%         +norm(VertStickM(:,3:end)*ParaPPCP,1)+ norm(HoriStickM(:,3:end)*ParaPPCP,1), opt);
%         +0.01*norm( RayMtilt*ParaPPCP,1)...        
%         +norm(spdiags(WeiCoP,0,size(CoPM1,1),size(CoPM1,1))*(CoPM1 - CoPM2)*ParaPPCP,1)...
%         +norm( (CoPM1 - CoPM2)*ParaPPCP,1 )...
ParaPPCP = double(ParaPPCP);
ParaPPCP = reshape(ParaPPCP,3,[]);
any(any(isnan(ParaPPCP)))

% porject the ray on planes to generate the ProjDepth
FitDepthPPCP = 80*ones(1,VertYNuDepth*HoriXNuDepth);
FitDepthPPCP(~maskSkyEroded) = (1./sum(ParaPPCP(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
FitDepthPPCP = reshape(FitDepthPPCP,VertYNuDepth,[]);
[Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
[VrmlName] = vrml_test_faceset_goodSkyBoundary(	filename{k}, Position3DFitedPPCP, FitDepthPPCP, permute(Ray,[2 3 1]), 'NoBLaser', ...
					[], [], 0, maskSkyEroded, maskG, 1, 0, a_default, b_default, Ox_default, Oy_default);
system(['gzip -9 -c ' ScratchDataFolder '/vrml/' VrmlName ' > ' ScratchDataFolder '/vrml/' VrmlName '.gz']);
delete([ScratchDataFolder '/vrml/' VrmlName]);
%return;
%=======================================================================================================================================
% Calculate Occlusion metrics
[SpatialJump] = DecideOcclusion(VertStickM,HoriStickM,ParaPPCP,ClosestNList);%[VertStickM(:,1:2) VertStickM(:,3:end)*ParaPPCP; HoriStickM(:,1:2) HoriStickM(:,3:end)*ParaPPCP];
OccluList = SpatialJump > log(threhJump);
[SpatialDisMask] = ShowCorner(MedSup,[ClosestNList OccluList]);

% CalCulate Corner Metrics
NormParaPPCP = ParaPPCP./repmat(norms(ParaPPCP),[3 1]);
ClosestNListCorner = [ClosestNList sum(abs(NormParaPPCP(:,Sup2Para(ClosestNList(:,1))) - NormParaPPCP(:,Sup2Para(ClosestNList(:,2)))),1)'];
CornerList = (ClosestNListCorner(:,3)/3)>threhConer ;
[CornerMask] = ShowCorner(MedSup,[ClosestNListCorner(:,1:2) CornerList ]);
save([ScratchDataFolder '/data/temp/List' num2str(k) '.mat'],'ClosestNList','ClosestNListCorner','CornerMask','SpatialJump','SpatialDisMask','OccluList','CornerList');

end
return;
