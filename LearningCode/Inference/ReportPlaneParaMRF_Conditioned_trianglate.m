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
function  [Position3DFited FitDepthPPCP] = ReportPlaneParaMRF_Conditioned_trianglate...
          (Default, Rotation, Translation, appendOpt, ...
          RayMatched, ClosestDepth, SupMatched, ...
          SampledGroundRayMatched, SampledGroundClosestDepth, SampledGroundSupMatched, ...
          DepthFolder,...
          Sup, SupOri, MedSup,depthMap,VarMap,RayOri, Ray, SupNeighborTable,...
          MedRay,maskSky,maskG,Algo,k, CornerList, OccluList,...
          MultiScaleSupTable, StraightLineTable, HBrokeBook, VBrokeBook,previoslyStored,...
          baseline, CoordinateFromRef, y0);
% This function runs the RMF over the plane parameter of each superpixels

if nargin <20
   baseline = 0;
end
if isempty(CoordinateFromRef)
   CoordinateFromRef = eye(3);
end
solverVerboseLevel = 0;

inferenceTime = tic;

fprintf(['\n     : Building Matrices....       ']);

% initialize parameters
GridFlag = 0;
NOYALMIP = 1;
NoSecondStep = 0;
Dual = false;
displayFlag = false;
RenderVrmlFlag = true;
PlaneParaSegmentationFlag = false;

scale = 1;
StickHori = 5;  %0.1; % sticking power in horizontal direction
StickVert = 10;     % sticking power in vertical direction
Center = 2; % Co-Planar weight at the Center of each superpixel
TriangulatedWeight = 20;
SampledGroundWeight = 10;
GroundLevelWright = 500;
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
WeiV = WeiV./sum(WeiV);% normalize if pair of superpixels have same index in all the scale, their weight will be 10
ShiftStick = -.1;  % between -1 and 0, more means more smoothing.
ShiftCoP = -.5;  % between -1 and 0, more means more smoothing.
gravity =true; % if true, apply the HoriConf and VertConf linear scale weight
CoPST = true; % if true, apply the Straight line prior as the Co-Planar constrain

%ConerImprove = false;
%FractionalDepthError = true;


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

   %NewMap = [rand(max(Sup(:)),3); [0 0 0]];
   
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
groundThreshold = cos([ zeros(1, Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)+10) ...
				linspace(0,15,ceil(Default.VertYNuDepth/2)-10)]*pi/180);
    %  v1 15 v2 20 too big v3 20 to ensure non misclassified as ground.
    %  verticalThreshold = cos(linspace(5,55,Default.VertYNuDepth)*pi/180); % give a vector of size 55 in top to down : 
verticalThreshold = cos([ 5*ones(1,Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)) ...
				linspace(5,55,ceil(Default.VertYNuDepth/2))]*pi/180); 
	% give a vector of size 55 in top to down : 
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
    YPointer = [YPointer; CenterY >= ceiling]; % Y is zero in the top ceiling default to be 0 as the top row in the image
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
      if any(CleanedDepthMap(mask) <=0)
         CleanedDepthMap(mask)
%         pause
      end
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

% =================Building up the triangulated  and sampled ground constrain ========================================
PosiTriangulatedM = sparse( 0, 3*NuSupSize);
count = 1;
for i = SupMatched'
    temp = sparse(1, 3*NuSupSize);
    if Sup2Para(i)*3>3*NuSupSize || Sup2Para(i)*3-2<0
       Sup2Para(i)
       NuSupSize
    end
    temp( (Sup2Para(i)*3-2):(Sup2Para(i)*3) ) = RayMatched(count,:)*ClosestDepth(count);
    PosiTriangulatedM = [PosiTriangulatedM; temp];
    count = count + 1;
end
PosiSampledGroundM = sparse( 0, 3*NuSupSize);
count = 1;
for i = SampledGroundSupMatched'
    temp = sparse(1, 3*NuSupSize);
    if (Sup2Para(i)*3)>(3*NuSupSize) || (Sup2Para(i)*3-2)<0
       Sup2Para(i)
       NuSupSize
    end
    temp( (Sup2Para(i)*3-2):(Sup2Para(i)*3) ) = SampledGroundRayMatched(count,:)*SampledGroundClosestDepth(count);
    PosiSampledGroundM = [PosiSampledGroundM; temp];
    count = count + 1;
end

% ================================================================================================

% buliding CoPlane Matrix=========================================================================

NuSTList = 0;
if CoPST
   NuSTList = size(CoPSTList,1);
   if ~isempty(CoPSTList)
      [V H] = size(SupNeighborTable);
      SupNeighborTable( CoPSTList(:,1)*V + CoPSTList(:,2)) = 1;
      SupNeighborTable( CoPSTList(:,2)*V + CoPSTList(:,1)) = 1;
%   ClosestNList = [ClosestNList; CoPSTList];
   end
end
CoPM1 = sparse(0,3*NuSupSize);
CoPM2 = sparse(0,3*NuSupSize);
CoPEstDepth = sparse(0,0);
CoPNorM = [];
WeiCoP = [];

for i = NuSup
    Neighbor = find( SupNeighborTable(i,:) ~=0);
    Neighbor = Neighbor( Neighbor> i);
    for j = Neighbor
        %  if ~CornerList(i)
        mask = Sup == i;
        SizeMaskAll = sum(mask(:));
        [y x] = find(mask);
        CenterX = round(median(x));
        CenterY = round(median(y));
        y = find(mask(:,CenterX));
        if ~isempty(y)
           CenterY = round(median(y));
        end
    
        temp1 = sparse(1, 3*NuSupSize);
        temp2 = sparse(1, 3*NuSupSize);
        temp1(:,(Sup2Para( i)*3-2): Sup2Para( i)*3) = ...
                                                   Ray(:,CenterY,CenterX)';
        temp2(:,(Sup2Para( j)*3-2): Sup2Para( j)*3) = ...
                                                   Ray(:,CenterY,CenterX)';
    %NuNei-NuSTList

%        if i < NuNei-NuSTList % only immediate connecting superpixels are weighted using MultiScaleSup
%       wei = WeiV*10;%*(MultiScaleSupTable(Sup2Para(ClosestNList(i,1)),2:end) == MultiScaleSupTable(Sup2Para(ClosestNList(i,2)),2:end))';  
           if MultiScaleFlag 
              vector = (MultiScaleSupTable(Sup2Para( i),2:end) == MultiScaleSupTable(Sup2Para( j),2:end));
              expV = exp(-10*(WeiV*vector' + ShiftCoP) );
              wei = 1/(1+expV);
           else
              wei = 1;
           end
%        else
%           wei = 1;
%        end
        oneRay1 = temp1*wei;
        oneRay2 = temp2*wei;
    %CoPM1 = [CoPM1; temp1*wei];
    %CoPM2 = [CoPM2; temp2*wei];

    
        tempWeiCoP = [SizeMaskAll];
        CoPEstDepth = [CoPEstDepth; max(median(CleanedDepthMap(mask)),ClosestDist)];
    
        mask = Sup == j;
        SizeMaskAll = sum(mask(:));
        [y x] = find(mask);
        CenterX = round(median(x));
        CenterY = round(median(y));
        y = find(mask(:,CenterX));
        if ~isempty(y)
           CenterY = round(median(y));
        end

        temp1 = sparse(1, 3*NuSupSize);
        temp2 = sparse(1, 3*NuSupSize);
        temp1(:,(Sup2Para( i)*3-2): Sup2Para( i)*3) = ... 
                                                            Ray(:,CenterY,CenterX)';
        temp2(:,(Sup2Para( j)*3-2): Sup2Para( j)*3) = ...
                                                            Ray(:,CenterY,CenterX)';
    
    % Instead of having separate L-1 terms for symmetric co-planar constraint; do the following:
    % If the penaly was ||.||_2^2 + ||.||_2^2; then the co-efficients are
    % some kind of average of two rays.  For one norm; we take its average.
    % (do not divide by 2 because the penalty should be double.
    
       CoPM1 = [CoPM1; temp1*wei + oneRay1 ];
       CoPM2 = [CoPM2; temp2*wei + oneRay2 ];
    
       tempWeiCoP = [tempWeiCoP; SizeMaskAll];
       WeiCoP = [WeiCoP; tempWeiCoP];
       CoPEstDepth = [CoPEstDepth; max(median(CleanedDepthMap(mask)),ClosestDist)];
  end
end%=========================================================================================================

%== find the boundary point that might need to be stick ot each other==========================================
HoriStickM_i = sparse(0,3*NuSupSize);
HoriStickM_j = sparse(0,3*NuSupSize);
HoriStickPointInd = [];
EstDepHoriStick = [];

MAX_POINTS_STITCH_HORI = 2;    % the actual code will be modified in another copy of the file
MIN_POINTS_STITCH = 2;      % ERROR:  not used.

% ================================================================
% NOTE: The actual algorithm should be picking precisely 2 points which are
% FARTHEST away from the candidate set of neighbors.  This algorithm
% will ALWAYS work and produce no surprising results.

% In some cases, one may experiment with picking only 1 point when the 2
% points are too close ---- this will make the algorithm faster; but might
% produce surprising (e.g. a triangle sticking out) sometimes.
% An ideal algorithm will reduce the number of points by checking for loops
% passing through 3 or less superpixels through this matrix; and removing
% them such that the smallest loop passes through 4 superpixels. (see EE263
% for a quick algorithm to do this -- involves product of matrices.
% ==================================================================

DIST_STICHING_THRESHOLD_HORI = 0.4;    
DIST_STICHING_THRESHOLD_HORI_ONLYCOL = -0.5;    % effectively not used, 

SupPixelNeighborList = sparse( max(Sup(:)), max(Sup(:)) );
SupPixelParsedList = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded1 = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded2 = sparse( max(Sup(:)), max(Sup(:)) );
addedIndexList = [ ];

BounaryPHori = conv2(Sup,[1 -1],'same') ~=0;
BounaryPHori(:,end) = 0;
BounaryPVert = conv2(Sup,[1; -1],'same') ~=0;
BounaryPVert(end,:) = 0;
boundariesHoriIndex = find(BounaryPHori==1)';
for i = boundariesHoriIndex
    j = i+Default.VertYNuDepth;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
    SupPixelParsedList(Sup(i),Sup(j)) = SupPixelParsedList(Sup(i),Sup(j)) + 1;
    
    if SupPixelNeighborList(Sup(i),Sup(j)) == 0
        recordAdded1(Sup(i),Sup(j)) = i;     
    elseif SupPixelNeighborList(Sup(i),Sup(j)) >= MAX_POINTS_STITCH_HORI
        continue;
    elseif SupPixelNeighborList(Sup(i),Sup(j)) == 1  % inside this remove the close stiching terms
        rowN = ceil(i/55);      colN = rem(i,55);
        rowN_older = ceil( recordAdded1(Sup(i),Sup(j)) / 55);
        colN_older = rem( recordAdded1(Sup(i),Sup(j)), 55);
        if abs(rowN - rowN_older) + (55/305)*abs(colN - colN_older) > DIST_STICHING_THRESHOLD_HORI && ...
            abs(colN - colN_older) > DIST_STICHING_THRESHOLD_HORI_ONLYCOL
            recordAdded2(Sup(i),Sup(j)) = i;
        else
            continue;
        end
    elseif SupPixelNeighborList(Sup(i),Sup(j)) == 2     %Assuming MAX_POINTS_STITCH = 3
        rowN = ceil(i/55);      colN = rem(i,55);
        rowN_older1 = ceil( recordAdded1(Sup(i),Sup(j)) / 55);
        colN_older1 = rem( recordAdded1(Sup(i),Sup(j)), 55);
        rowN_older2 = ceil( recordAdded2(Sup(i),Sup(j)) / 55);
        colN_older2 = rem( recordAdded2(Sup(i),Sup(j)), 55);
        
        if abs(rowN - rowN_older1) + (55/305)*abs(colN - colN_older1) > DIST_STICHING_THRESHOLD_HORI && ...
            abs(rowN - rowN_older2) + (55/305)*abs(colN - colN_older2) > DIST_STICHING_THRESHOLD_HORI
            ;
        else
            continue;
        end   
    end

    % If you come here, it means you are probably adding it.
    SupPixelNeighborList(Sup(i),Sup(j)) = SupPixelNeighborList(Sup(i),Sup(j)) + 1;
    addedIndexList = [addedIndexList i];
end


WeightHoriNeighborStitch = [ ];

for i = addedIndexList
    j = i+Default.VertYNuDepth;
   
    WeightHoriNeighborStitch = [WeightHoriNeighborStitch;  SupPixelParsedList(Sup(i),Sup(j)) / ...
                                            SupPixelNeighborList(Sup(i),Sup(j)) ];

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
% ==============================================



% ======== finding the unucessary stiching points in Vertical direction ====
VertStickM_i = sparse(0,3*NuSupSize);
VertStickM_j = sparse(0,3*NuSupSize);
VertStickPointInd = [];
EstDepVertStick = [];

MAX_POINTS_STITCH_VERT = 4;	%3
DIST_STICHING_THRESHOLD_VERT = 0.1;	%0.3 
DIST_STICHING_THRESHOLD_VERT_ONLYCOL = -0.5;    % effectively not used, ideally should be 0.5; i.e., the point should be farther in col direction because that is the direction of the edge.

SupPixelNeighborList = sparse( max(Sup(:)), max(Sup(:)) );
SupPixelParsedList = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded1 = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded2 = sparse( max(Sup(:)), max(Sup(:)) );
addedIndexList = [ ];

for i = find(BounaryPVert==1)'
    j = i+1;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
    SupPixelParsedList(Sup(i),Sup(j)) = SupPixelParsedList(Sup(i),Sup(j)) + 1;
    
    if SupPixelNeighborList(Sup(i),Sup(j)) == 0
        recordAdded1(Sup(i),Sup(j)) = i;
    elseif SupPixelNeighborList(Sup(i),Sup(j)) >= MAX_POINTS_STITCH_VERT
        continue;
    elseif SupPixelNeighborList(Sup(i),Sup(j)) == 1  % inside this remove the close stiching terms
        rowN = ceil(i/55);      colN = rem(i,55);
        rowN_older = ceil( recordAdded1(Sup(i),Sup(j)) / 55);
        colN_older = rem( recordAdded1(Sup(i),Sup(j)), 55);
        if abs(rowN - rowN_older) + (55/305)*abs(colN - colN_older) > DIST_STICHING_THRESHOLD_VERT && ...
                 abs(colN - colN_older) > DIST_STICHING_THRESHOLD_VERT_ONLYCOL
            recordAdded2(Sup(i),Sup(j)) = i;
        else
            continue;
        end
    elseif SupPixelNeighborList(Sup(i),Sup(j)) == 2     %Assuming MAX_POINTS_STITCH = 3
        rowN = ceil(i/55);      colN = rem(i,55);
        rowN_older1 = ceil( recordAdded1(Sup(i),Sup(j)) / 55);
        colN_older1 = rem( recordAdded1(Sup(i),Sup(j)), 55);
        rowN_older2 = ceil( recordAdded2(Sup(i),Sup(j)) / 55);
        colN_older2 = rem( recordAdded2(Sup(i),Sup(j)), 55);
        
        if abs(rowN - rowN_older1) + (55/305)*abs(colN - colN_older1) > DIST_STICHING_THRESHOLD_VERT && ...
            abs(rowN - rowN_older2) + (55/305)*abs(colN - colN_older2) > DIST_STICHING_THRESHOLD_VERT
            ;
        else
            continue;
        end        
     end

    % If you come here, it means you are probably adding it.
    SupPixelNeighborList(Sup(i),Sup(j)) = SupPixelNeighborList(Sup(i),Sup(j)) + 1;
    addedIndexList = [addedIndexList i];
end


WeightVertNeighborStitch = [ ];

for i = addedIndexList
    j = i+1;
   
     WeightVertNeighborStitch = [WeightVertNeighborStitch;  SupPixelParsedList(Sup(i),Sup(j)) / ...
                                            SupPixelNeighborList(Sup(i),Sup(j)) ];

    Target(1) = Sup2Para(Sup(i));
    Target(2) = Sup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
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

end
% ====finished finding the unucessary stiching points in Vertical direction ===


% ======================================Finish building up matrix=====================hard work======================



% ================================================================================================================
depthfile = strrep(Default.filename{k},'img','depth_learned'); %


% Start Decompose the image align with superpixels ==================================================================================
% define the decomposition in X direction only

% ============== parameters for the decomposition problem
% Optimal parameters for current code:
% For one machine: Sedumi: (1,1)        Lpsolve:(3,1)
% Multiple machines: Sedumi: (4,1)      LPsolve:(3,1)
% lpsolve running time is 22 seconds for (4,2) arrangement; but numerical
% accuracy needs to be resolved first.
XNuDecompose = 2;       
YNuDecompose = 1;
% ============ parameters for the decomposition problem


 
TotalRectX = 2*XNuDecompose-1;
TotalRectY= 2*YNuDecompose-1;
PlanePara = NaN*ones(3*NuSupSize,1); % setup the lookuptable for the solved plane parameter

 opt = sdpsettings('solver','sedumi','cachesolvers',1,'sedumi.eps',1e-6,'verbose',solverVerboseLevel);
%     opt = sdpsettings('solver','lpsolve','cachesolvers',1,'verbose',5);
%     opt = sdpsettings('solver','lpsolve','cachesolvers',1,'showprogress',1, 'verbose',4);
%    opt = sdpsettings('solver','glpk','cachesolvers',1, 'usex0', 1);    
%    opt = sdpsettings('solver','glpk', 'usex0', 1);    
    %opt = sdpsettings('solver','linprog','cachesolvers',1);    
%    opt = sdpsettings('solver','bpmpd');
    
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
    SubSup = sort(setdiff(unique( reshape( Sup(RangeY,RangeX),1,[])),0));
    BoundarySup = [];
%    for m = SubSup
    if any(SubSup <=0)
       SubSup(SubSup<=0)
    end
    BoundarySup = find(sum(SupNeighborTable(SubSup,:), 1) ~=0);
    BoundarySup = unique(setdiff(BoundarySup,[0 SubSup] ));

    % chech if BoundarySup non-NaN in PlanePara
    checkNoNNaN = ~isnan(PlanePara(Sup2Para(BoundarySup)*3));
    BoundarySup = BoundarySup(checkNoNNaN);
    TotalSup = sort([SubSup BoundarySup]);

    SubSupPtr = [ Sup2Para(SubSup)*3-2;...
                    Sup2Para(SubSup)*3-1;...
                    Sup2Para(SubSup)*3];
    SubSupPtr = SubSupPtr(:);
    BoundarySupPtr = [ Sup2Para(BoundarySup)*3-2;...
                       Sup2Para(BoundarySup)*3-1;...
                       Sup2Para(BoundarySup)*3];
    BoundarySupPtr = BoundarySupPtr(:);
    NuSubSupSize = size(SubSup,2);
    NewRayAllM = RayAllM(:,SubSupPtr);
    tar = sum(NewRayAllM ~= 0,2) == 3;
    NewRayAllM = NewRayAllM(tar,:);
    NewPosiM = PosiM(:,SubSupPtr);
    tar = sum(NewPosiM ~= 0,2) == 3;
    NewPosiM = NewPosiM(tar,:);
    NewVarM = VarM(tar);
    NewPosiTriangulatedM = PosiTriangulatedM(:,SubSupPtr);
    tar = sum(NewPosiTriangulatedM ~= 0,2) == 3;
    NewPosiTriangulatedM = NewPosiTriangulatedM(tar,:);
    NewPosiSampledGroundM = PosiSampledGroundM(:,SubSupPtr);
    tar = sum(NewPosiSampledGroundM ~= 0,2) == 3;
    NewPosiSampledGroundM = NewPosiSampledGroundM(tar,:);
    
    NewCoPM = CoPM1(:,SubSupPtr) - CoPM2(:,SubSupPtr);
    NewCoPMBound = CoPM1(:,BoundarySupPtr) - CoPM2(:,BoundarySupPtr);
    tar = sum(NewCoPM ~= 0,2) + sum(NewCoPMBound ~= 0,2)==6;
    NewCoPMBound = NewCoPMBound*PlanePara(BoundarySupPtr);
    NewCoPM = NewCoPM(tar,:);
    NewCoPMBound = NewCoPMBound(tar);
    NewCoPEstDepth = CoPEstDepth(tar);   

    NewHoriStickM = HoriStickM_i(:,SubSupPtr)-HoriStickM_j(:,SubSupPtr);
    NewHoriStickMBound = HoriStickM_i(:,BoundarySupPtr)-HoriStickM_j(:,BoundarySupPtr);
    tar = sum(NewHoriStickM ~= 0,2)+  sum(NewHoriStickMBound ~= 0,2) ==6;
    NewHoriStickM = NewHoriStickM(tar,:);    
    NewEstDepHoriStick = EstDepHoriStick(tar);
    NewHoriStickMBound = NewHoriStickMBound*PlanePara(BoundarySupPtr);
    NewHoriStickMBound = NewHoriStickMBound(tar);
	NewWeightHoriNeighborStitch = WeightHoriNeighborStitch(tar);

    NewVertStickM = VertStickM_i(:,SubSupPtr)-VertStickM_j(:,SubSupPtr);
    NewVertStickMBound = VertStickM_i(:,BoundarySupPtr)-VertStickM_j(:,BoundarySupPtr);
    tar = sum(NewVertStickM ~= 0,2) + sum(NewVertStickMBound ~= 0,2)==6;
    NewVertStickM = NewVertStickM(tar,:);
    NewEstDepVertStick = EstDepVertStick(tar);
    NewVertStickMBound = NewVertStickMBound*PlanePara(BoundarySupPtr);
    NewVertStickMBound = NewVertStickMBound(tar);
	NewWeightVertNeighborStitch = WeightVertNeighborStitch(tar);
 
        ParaPPCP = sdpvar(3*NuSubSupSize,1);
        F = set(ParaPPCP(3*(1:NuSubSupSize)-1).*YPointer(Sup2Para(SubSup))<=0)...
            +set(NewRayAllM*ParaPPCP <=1/ClosestDist)...
            +set(NewRayAllM*ParaPPCP >=1/FarestDist);

% ================================================================

	WeightsSelfTerm = 1 ./ exp(abs(NewVarM)/BandWith);
    
% ============ solver specific options ======
   if strcmp(opt.solver, 'glpk')
      disp('Using x0 for GPLK');
      ParaPPCP = NewPosiM \ ones(size(NewPosiM,1),1);
   end
   fprintf(['     ' num2str( toc(inferenceTime) ) '\n     : In 1st level Optimization, using ' opt.solver '.  ' ...
            '(' num2str(k+1) '/' num2str((TotalRectX-1)+1) ',' num2str(l+1) '/' num2str((TotalRectY-1)+1) ')']);

          sol = solvesdp(F, norm( (NewPosiM*ParaPPCP-ones(size(NewPosiM,1),1)) .* WeightsSelfTerm,1)...
                        + TriangulatedWeight*norm( NewPosiTriangulatedM*ParaPPCP-ones(size(NewPosiTriangulatedM,1), 1), 1)+...
                        + SampledGroundWeight*norm( NewPosiSampledGroundM*ParaPPCP-ones(size(NewPosiSampledGroundM,1), 1), 1)+...
              		+ Center*norm(((NewCoPM)*ParaPPCP + NewCoPMBound).*NewCoPEstDepth, 1)...
              		+ norm(((NewHoriStickM)*ParaPPCP + NewHoriStickMBound).*...
                                 NewEstDepHoriStick.*NewWeightHoriNeighborStitch,1)...
                 	+ norm(((NewVertStickM)*ParaPPCP + NewVertStickMBound).*...
                                 NewEstDepVertStick.*NewWeightVertNeighborStitch,1) ...
	                , opt);

    ParaPPCP = double(ParaPPCP);
%     sum(isnan(ParaPPCP))
    yalmip('clear');
    PlanePara(SubSupPtr) = ParaPPCP;

  end
end

% build the whole image
    PlanePara = reshape(PlanePara,3,[]);
    %any(any(isnan(PlanePara)))
    % porject the ray on planes to generate the ProjDepth
    FitDepthPPCP = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
    FitDepthPPCP(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded ))).*Ray(:,~maskSkyEroded ),1))';
    FitDepthPPCP = reshape(FitDepthPPCP,Default.VertYNuDepth,[]);

    % =====================================================
    [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
    if NoSecondStep %RenderVrmlFlag && i==3
         [a b c] = size(Position3DFitedPPCP);
         % ============transfer to world coordinate
         Position3DFitedPPCP = Rotation*Position3DFitedPPCP(:,:) + repmat(Translation, 1, length(Position3DFitedPPCP(:,:)));
         Position3DFitedPPCP = reshape(Position3DFitedPPCP, a, b, []);
         % ========================================
         Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
         Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
         RR =permute(Ray,[2 3 1]);
         temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
         PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),[Default.VertYNuDepth Default.HoriXNuDepth 1])+repmat(cat(3,Default.Ox_default,Default.Oy_default),[Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
         PositionTex = permute(PositionTex,[2 3 1]);
         WrlFacestHroiReduce(Position3DFitedPPCP,PositionTex,SupOri, [ Default.filename{1} '1st'],[ Default.Wrlname{1} '1st'], ...
                             Default.OutPutFolder, GridFlag, appendOpt);
         system(['gzip -9 -c ' Default.OutPutFolder Default.filename{1} '1st.wrl > ' ...
                Default.OutPutFolder Default.filename{1} '1st.wrl.gz']);
         system(['cp '  Default.OutPutFolder Default.filename{1} '1st.wrl.gz ' ...
                Default.OutPutFolder Default.filename{1} '1st.wrl']);
         delete([Default.OutPutFolder Default.filename{1} '1st.wrl.gz']);
    end

    % Trial for segmentaion of the plane parameters
    if PlaneParaSegmentationFlag
       TempSup = Sup;%imresize(Sup,[330 610]);
       TempSup(TempSup==0) = max(TempSup(:))+1;
       TempSup2Para = Sup2Para;
       TempSup2Para(end+1) = size(PlanePara,2);
       TempPlanePara = PlanePara;
       TempPlanePara(:,end+1) = 0;
       PlaneParaPics = TempPlanePara(:,TempSup2Para( TempSup));
       PlaneParaPics = PlaneParaPics./repmat( 2*max(abs(PlaneParaPics),[],2), 1, size(PlaneParaPics,2));
       PlaneParaPics = permute( reshape( PlaneParaPics, 3, 55, []), [2 3 1])+0.5;
       MergedPlaneParaPics = segmentImgOpt( Default.sigm*scale, Default.k*scale, Default.minp*10*scale, PlaneParaPics, '', 0) + 1;
       if Default.Flag.DisplayFlag
          figure(400);
          imaegsc( PlaneParaPics);
          imaegsc( MergedPlaneParaPics);
       end
       save('/afs/cs/group/reconstruction3d/scratch/testE/PlaneParaPics.mat','PlaneParaPics','MergedPlaneParaPics');
    end

%return;
%==================Finished for one step MRF==========================================================================================================

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
  %groundPara = abs(normalizedPara(2,:)) >= groundThreshold(YPosition); % assuming ground has [0; 1; 0] surface norm
  groundPara = abs(CoordinateFromRef(:,2)'*normalizedPara) >= groundThreshold(YPosition); % CoordinateFromRef(:,2) is the new ground surface norm
  groundParaInd = find(groundPara);
%  verticalPara = abs(normalizedPara(2,:)) <= verticalThreshold(YPosition); % ssuming ground has [0; 1; 0] surface norm
  verticalPara = abs( CoordinateFromRef(:,2)'*normalizedPara) <= verticalThreshold(YPosition); % CoordinateFromRef(:,2) is the new ground surface norm
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

  opt = sdpsettings('solver','sedumi','cachesolvers',1, 'verbose', solverVerboseLevel);
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
%       SubSup = sort(setdiff(reshape( Sup(RangeY,RangeX),[],1),0))';
      SubSup = sort(setdiff( unique( reshape( Sup(RangeY,RangeX),1,[])),0));
      BoundarySup = [];
%      for m = SubSup
      BoundarySup = find(sum(SupNeighborTable(SubSup,:), 1) ~=0);
%          maskList = ClosestNList(:,1) == m;
%          BoundarySup = [ BoundarySup ClosestNList(maskList,2)'];
%      end
      BoundarySup = unique(setdiff(BoundarySup,[0 SubSup] ));

      % chech if BoundarySup non-NaN in PlanePara
      checkNoNNaN = ~isnan(PlanePara(Sup2Para(BoundarySup)*3));
      BoundarySup = BoundarySup(checkNoNNaN);
      TotalSup = sort([SubSup BoundarySup]);

%      TotalSupPtr = [ Sup2Para(TotalSup)*3-2;...
%                      Sup2Para(TotalSup)*3-1;...
%                      Sup2Para(TotalSup)*3];
%      TotalSupPtr = TotalSupPtr(:);
      SubSupPtr = [ Sup2Para(SubSup)*3-2;...
                    Sup2Para(SubSup)*3-1;...
                    Sup2Para(SubSup)*3];
      SubSupPtr = SubSupPtr(:);
      BoundarySupPtr = [ Sup2Para(BoundarySup)*3-2;...
                         Sup2Para(BoundarySup)*3-1;...
                         Sup2Para(BoundarySup)*3];
      BoundarySupPtr =BoundarySupPtr(:);
%      NuSubSupSize = size(TotalSup,2);
      NuSubSupSize = size(SubSup,2);
%      TotalSup2Para = sparse(1,max(TotalSup));
%      TotalSup2Para(TotalSup) = 1:NuSubSupSize;
      SubSup2Para = sparse(1,max(SubSup));
      SubSup2Para(SubSup) = 1:NuSubSupSize;
%      BoundarySupPtrSub = [ TotalSup2Para(BoundarySup)*3-2;...
%                            TotalSup2Para(BoundarySup)*3-1;...
%                            TotalSup2Para(BoundarySup)*3];
%      BoundarySupPtrSub =BoundarySupPtrSub(:);
%      SubSupPtrSub = [ TotalSup2Para(SubSup)*3-2;...
%                       TotalSup2Para(SubSup)*3-1;...
%                       TotalSup2Para(SubSup)*3];
%      SubSupPtrSub =SubSupPtrSub(:);
    
      % clearn RayAllM PosiM CoPM1 HoriStickM_i VertStickM_i
%      NewRayAllM = RayAllM(:,TotalSupPtr);
      NewRayAllM = RayAllM(:,SubSupPtr);
      tar = sum(NewRayAllM ~= 0,2) ==3;
      NewRayAllM = NewRayAllM(tar,:);

%      NewPosiMPPCP = PosiMPPCP(:,TotalSupPtr);
      NewPosiMPPCP = PosiMPPCP(:,SubSupPtr);
      tar = sum(NewPosiMPPCP ~= 0,2) ==3;
      NewPosiMPPCP = NewPosiMPPCP(tar,:);
      NewVarM = VarM2(tar);
      NewPosiTriangulatedM = PosiTriangulatedM(:,SubSupPtr);
      tar = sum(NewPosiTriangulatedM ~= 0,2) == 3;
      NewPosiTriangulatedM = NewPosiTriangulatedM(tar,:);
      NewPosiSampledGroundM = PosiSampledGroundM(:,SubSupPtr);
      tar = sum(NewPosiSampledGroundM ~= 0,2) == 3;
      NewPosiSampledGroundM = NewPosiSampledGroundM(tar,:);
    
%      NewCoPM = CoPM1(:,TotalSupPtr) - CoPM2(:,TotalSupPtr);
      NewCoPM = CoPM1(:,SubSupPtr) - CoPM2(:,SubSupPtr);
      NewCoPMBound = CoPM1(:,BoundarySupPtr) - CoPM2(:,BoundarySupPtr);
%      tar = sum( NewCoPM ~= 0,2)  ==6;
      tar = sum( NewCoPM ~= 0,2) + sum( NewCoPMBound ~= 0,2) ==6;
      NewCoPM = NewCoPM(tar,:);
      NewCoPMBound = NewCoPMBound*PlanePara(BoundarySupPtr); % column vertor
      NewCoPMBound = NewCoPMBound(tar);
      NewCoPEstDepth = CoPEstDepth(tar);   

%      NewHoriStickM = HoriStickM_i(:,TotalSupPtr)-HoriStickM_j(:,TotalSupPtr);
      NewHoriStickM = HoriStickM_i(:,SubSupPtr)-HoriStickM_j(:,SubSupPtr);
      NewHoriStickMBound = HoriStickM_i(:,BoundarySupPtr)-HoriStickM_j(:,BoundarySupPtr);
%      tar = sum(NewHoriStickM ~= 0,2) ==6;
      tar = sum(NewHoriStickM ~= 0,2) + sum( NewHoriStickMBound ~= 0,2)==6;
      NewHoriStickM = NewHoriStickM(tar,:);
      NewHoriStickMBound = NewHoriStickMBound*PlanePara(BoundarySupPtr); % column vertor
      NewHoriStickMBound = NewHoriStickMBound(tar);
      NewEstDepHoriStick = EstDepHoriStick(tar);
         NewWeightHoriNeighborStitch = WeightHoriNeighborStitch(tar);

%      NewVertStickM = VertStickM_i(:,TotalSupPtr)-VertStickM_j(:,TotalSupPtr);
      NewVertStickM = VertStickM_i(:,SubSupPtr)-VertStickM_j(:,SubSupPtr);
      NewVertStickMBound = VertStickM_i(:, BoundarySupPtr)-VertStickM_j(:,BoundarySupPtr);
%      tar = sum(NewVertStickM ~= 0,2) ==6;
      tar = sum(NewVertStickM ~= 0,2) + sum(NewVertStickMBound ~= 0,2)==6;
      NewVertStickM = NewVertStickM(tar,:);
      NewVertStickMBound = NewVertStickMBound*PlanePara(BoundarySupPtr); % column vertor
      NewVertStickMBound = NewVertStickMBound(tar);
      NewEstDepVertStick = EstDepVertStick(tar);
         NewWeightVertNeighborStitch = WeightVertNeighborStitch(tar);

%    try reduce the vertical constrain
     %NonVertPtr = setdiff( 1:(3*NuSubSupSize), SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))*3-1 );
     YNoComingBack = YPointer(Sup2Para(SubSup));
%     YNoComingBack(SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))) = [];
%     YCompMask = zeros(1,3*NuSubSupSize);
%     YCompMask(3*(1:NuSubSupSize)-1) = 1;
%     YCompMask = YCompMask(NonVertPtr); 
     XCompMask = zeros(1,3*NuSubSupSize);
%     XCompMask( SubSup2Para( NuSup( (intersect( indexGroundX,SubSupPtr ) +2)/3))*3-2 ) = 1;
     XCompMask = SubSup2Para( NuSup( (intersect( indexGroundX,SubSupPtr ) +2)/3));
     VertMask = intersect( find(verticalPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) ;
%     XCompMask = XCompMask(NonVertPtr);
     YCompMask = zeros(1,3*NuSubSupSize);
     YCompMask = SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr )+1 )/3));
%     ZCompMask = ZCompMask(NonVertPtr);
     GroundMask = intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) ;

      Para = sdpvar( 3*NuSubSupSize,1);
%      F = set(Para(3*(1:NuSubSupSize)-1).*YPointer(Sup2Para(SubSup))<=0)...
%          +set( Para( SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))*3-1) == 0); 
%          +set(Para(BoundarySupPtrSub) == PlanePara(BoundarySupPtr) )...
          % Special constrain for decomp fix the solved neighbor plane parameter       

          %      +set(Para(indexGroundX) == 0)...
          %      +set(Para(indexGroundZ) == 0);

%       if FractionalDepthError
% %         size(PosiMPPCP)

    fprintf([ '     ' num2str( toc(inferenceTime) ) '\n     : In 2nd level Optimization, using ' opt.solver '.  ' ...
                     '(' num2str(k+1) '/' num2str((TotalRectX-1)+1) ',' num2str(l+1) '/' num2str((TotalRectY-1)+1) ')']);
    if ~isempty(y0)    
      disp('y0 constrain');
      F = set( CoordinateFromRef(:,2)'*reshape(Para,3,[]).*YNoComingBack'<=0)...
          +set(NewRayAllM*Para <=1/ClosestDist)...
          +set(NewRayAllM*Para >=1/FarestDist)...
          ;
%          +set( Para( XCompMask*3-2) == Rotation(2,1)'./(y0 - Translation(2)) )...
%          +set( Para( XCompMask*3-1) == Rotation(2,2)'./(y0 - Translation(2)) )...
%          +set( Para( XCompMask*3) == Rotation(2,3)'./(y0 - Translation(2)) )...
       sol=solvesdp(F,norm( ( NewPosiMPPCP*Para-ones(size(NewPosiMPPCP,1),1))./exp(abs(NewVarM)/BandWith),1)...
                   +TriangulatedWeight*norm( NewPosiTriangulatedM*Para-ones(size(NewPosiTriangulatedM,1), 1), 1)+...
                   +SampledGroundWeight*norm( NewPosiSampledGroundM*Para-ones(size(NewPosiSampledGroundM,1), 1), 1)+...
                   +Center*norm(( NewCoPM*Para + NewCoPMBound).*NewCoPEstDepth, 1)...
                   +norm(( NewHoriStickM*Para + NewHoriStickMBound).*...
                           NewEstDepHoriStick.*NewWeightHoriNeighborStitch,1)...
                   +norm(( NewVertStickM*Para + NewVertStickMBound).*...
                           NewEstDepVertStick.*NewWeightVertNeighborStitch,1)...
                   +10*norm( ( CoordinateFromRef(:,1)'*reshape( Para( reshape( [XCompMask*3-2; XCompMask*3-1 ;XCompMask*3], [], 1) ) , 3, []) )./...
                              normPara( GroundMask ), 1)...
                   +10*norm( ( CoordinateFromRef(:,3)'*reshape( Para( reshape( [XCompMask*3-2; XCompMask*3-1 ;XCompMask*3], [], 1) ) , 3, []))./...
                              normPara( GroundMask ), 1) ...
                   +10*norm( ( CoordinateFromRef(:,2)'*reshape( Para( reshape( [YCompMask*3-2; YCompMask*3-1 ;YCompMask*3], [], 1) ) , 3, []))./...
                              normPara( VertMask ), 1) ...
                   +GroundLevelWright*norm( Para( XCompMask*3-2) - Rotation(2,1)'./(y0 - Translation(2)), 1)...
                   +GroundLevelWright*norm( Para( XCompMask*3-1) - Rotation(2,2)'./(y0 - Translation(2)), 1)...
                   +GroundLevelWright*norm( Para( XCompMask*3) - Rotation(2,3)'./(y0 - Translation(2)), 1)...
                   , opt);
    else
      F = set( CoordinateFromRef(:,2)'*reshape(Para,3,[]).*YNoComingBack'<=0)...
          +set(NewRayAllM*Para <=1/ClosestDist)...
          +set(NewRayAllM*Para >=1/FarestDist);
       sol=solvesdp(F,norm( ( NewPosiMPPCP*Para-ones(size(NewPosiMPPCP,1),1))./exp(abs(NewVarM)/BandWith),1)...
                   +TriangulatedWeight*norm( NewPosiTriangulatedM*Para-ones(size(NewPosiTriangulatedM,1), 1), 1)+...
                   +SampledGroundWeight*norm( NewPosiSampledGroundM*Para-ones(size(NewPosiSampledGroundM,1), 1), 1)+...
                   +Center*norm(( NewCoPM*Para + NewCoPMBound).*NewCoPEstDepth, 1)...
                   +norm(( NewHoriStickM*Para + NewHoriStickMBound).*...
                           NewEstDepHoriStick.*NewWeightHoriNeighborStitch,1)...
                   +norm(( NewVertStickM*Para + NewVertStickMBound).*...
                           NewEstDepVertStick.*NewWeightVertNeighborStitch,1)...
                   +10*norm( ( CoordinateFromRef(:,1)'*reshape( Para( reshape( [XCompMask*3-2; XCompMask*3-1 ;XCompMask*3], [], 1) ) , 3, []) )./...
                              normPara( GroundMask ), 1)...
                   +10*norm( ( CoordinateFromRef(:,3)'*reshape( Para( reshape( [XCompMask*3-2; XCompMask*3-1 ;XCompMask*3], [], 1) ) , 3, []))./...
                              normPara( GroundMask ), 1) ...
                   +10*norm( ( CoordinateFromRef(:,2)'*reshape( Para( reshape( [YCompMask*3-2; YCompMask*3-1 ;YCompMask*3], [], 1) ) , 3, []))./...
                              normPara( VertMask ), 1) ...
                   , opt);
    end
% ==============not used ============
%                 +10*norm( ( Para( XCompMask ))./...
%                            normPara( intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) )', 1)...
%                 +10*norm( ( Para( ZCompMask))./...
%                            normPara( intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) )', 1) ...

             
             
             %             sqrt( max(CoPM1*ParaPPCP(:), 1/FarestDist).*max(CoPM2*ParaPPCP(:), 1/FarestDist)), 1)...
%             +norm(((VertStickM_i-VertStickM_j)*Para)./...
%             sqrt( max(VertStickM_i*ParaPPCP(:), 1/FarestDist).*max(VertStickM_j*ParaPPCP(:), 1/FarestDist)),1)...
%   pause;
    %         +1000*norm( Para(indexVertical)./ normPara( find(verticalPara) )',1) ...
%            +norm(((HoriStickM_i-HoriStickM_j)*Para)./sqrt((VertStickM_i*ParaPPCP(:)).*(VertStickM_j*ParaPPCP(:))),1)...
% %       else % not used anymore
%          solvesdp(F,norm( RayMd*Para - DepthInverseM,1)... 
%                  +Center*norm((CoPM1 - CoPM2)*Para,1)...
%                  +norm((VertStickM_i-VertStickM_j)*Para,1)...
%                  +norm((HoriStickM_i-HoriStickM_j)*Para,1)...
%                  +1000*norm( Para(indexVertical)./ normPara( find(verticalPara) )',1) ...
%                  +1000*norm( Para(indexGroundX)./ normPara( find(groundPara) )', 1)  ...
%                  +1000*norm( Para(indexGroundZ)./ normPara( find(groundPara) )', 1) ...
%                  , opt);
% %         +0.01*norm( RayMtilt*ParaPPCP,1)...        
% %         +norm(spdiags(WeiCoP,0,size(CoPM1,1),size(CoPM1,1))*(CoPM1 - CoPM2)*ParaPPCP,1)...
% %         +norm( (CoPM1 - CoPM2)*ParaPPCP,1 )...
%      end
% ==================================
     Para = double(Para);
     %sum(isnan(Para))
     yalmip('clear');
     tempPara = zeros(3*NuSubSupSize,1);
     tempPara = Para;     
%      tempPara(NonVertPtr) = Para;
%     PlanePara(TotalSupPtr) = Para;
     PlanePara(SubSupPtr) = tempPara;
    end
  end

  %pause
  PlanePara = reshape(PlanePara,3,[]);
  FitDepth = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
  FitDepth(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
  FitDepth = reshape(FitDepth,Default.VertYNuDepth,[]);
  
  % ==========Storage ==============
  if Default.Flag.AfterInferenceStorage
     save([ Default.ScratchFolder '/' strrep( Default.filename{1},'.jpg','') '_AInf.mat' ], 'FitDepth', ...
            'Sup', 'SupOri', 'MedSup', 'RayOri','Ray','SupNeighborTable','maskSky','maskG','MultiScaleSupTable');
  end
  % ===============================
  %sum(isnan(FitDepth(:)))
  [Position3DFited] = im_cr2w_cr(FitDepth,permute(Ray,[2 3 1]));
  [a b c] = size(Position3DFited);
  % ============transfer to world coordinate
  Position3DFited = Rotation*Position3DFited(:,:) + repmat(Translation, 1, length(Position3DFited(:,:)));
  Position3DFited = reshape(Position3DFited, a, b, []);
  % ========================================
  Position3DFited(3,:) = -Position3DFited(3,:);
  Position3DFited = permute(Position3DFited,[2 3 1]);
  RR =permute(Ray,[2 3 1]);
  temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
  PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),...
                [Default.VertYNuDepth Default.HoriXNuDepth 1])...
                +repmat(cat(3,Default.Ox_default,Default.Oy_default),...
                [Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
  PositionTex = permute(PositionTex,[2 3 1]);


  fprintf(['     ' num2str( toc(inferenceTime) ) '\n     : Writing WRL.']);
  WrlFacestHroiReduce(Position3DFited, PositionTex, SupOri, Default.filename{1}, Default.Wrlname{1}, ...
      Default.OutPutFolder, GridFlag, appendOpt);
  
%   system(['gzip -9 -c ' Default.OutPutFolder Default.Wrlname{1} '.wrl > ' ...
%          Default.OutPutFolder Default.Wrlname{1} '.wrl.gz']);
%   system(['cp '  Default.OutPutFolder Default.Wrlname{1} '.wrl.gz ' ...
%           Default.OutPutFolder Default.Wrlname{1} '.wrl']);
%   delete([Default.OutPutFolder Default.Wrlname{1} '.wrl.gz']);

    % Trial for segmentaion of the plane parameters
    if PlaneParaSegmentationFlag
       TempSup = Sup;%imresize(Sup,[330 610]);
       TempSup(TempSup==0) = max(TempSup(:))+1;
       TempSup2Para = Sup2Para;
       TempSup2Para(end+1) = size(PlanePara,2);
       TempPlanePara = PlanePara;
       TempPlanePara(:,end+1) = 0;
       PlaneParaPics = TempPlanePara(:,TempSup2Para( TempSup));
       PlaneParaPics = PlaneParaPics./repmat( 2*max(abs(PlaneParaPics),[],2), 1, size(PlaneParaPics,2));
       PlaneParaPics2 = permute( reshape( PlaneParaPics, 3, 55, []), [2 3 1])+0.5;
       MergedPlaneParaPics = segmentImgOpt( Default.sigm*scale, Default.k*scale, Default.minp*10*scale, PlaneParaPics, '', 0) + 1;
       if Default.Flag.DisplayFlag
          figure(400);
          imaegsc( PlaneParaPics);
          imaegsc( MergedPlaneParaPics);
       end
       save('/afs/cs/group/reconstruction3d/scratch/testE/PlaneParaPics2.mat','PlaneParaPics2','MergedPlaneParaPics');
    end
% save depth Map ++++++++++++++++
%       depthMap = FitDepth;
%       system(['mkdir ' Default.ScratchDataFolder '/VNonSupport_' DepthFolder '/']);
%       save([Default.ScratchDataFolder '/VNonSupport_' DepthFolder '/' depthfile '.mat'],'depthMap');
% =============================
% return;
