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
function  ReportPlaneParaMRF_Conditioned(Default, Sup, SupOri,depthMap,VarMapRaw, ...
	  	RayOri, Ray, SupNeighborTable, maskSky,maskG, MultiScaleSupTable, ...
		StraightLineTable);
% This function runs the RMF over the plane parameter of each superpixels

% Input:
% Default- all default parameters
% Sup - Superpixel, in 2-d matrix. 
% SupOri - original superpixel before cleaning
% DepthMap - DepthMap from Learning or coarse measuring
% VarMap - confidance for DepthMap (latest algo VarMap used simply one since the learning not generalized well)
% RayOri - Rays without stiching to the boundary.
% Ray - rays after stiching
% SupNeighborTable - A lookup table for superpixels' neighbors (2-d sparse matrix)
% maskSky - Sky mask
% maskG - Ground mask
% MultiScaleSupTable - multiple scale segmentation, used to define the weights betwenn the stiching terms.
% StraightLineTable - straight line stiching, (not used in this version, but might be very usefulif implemented).

% ------------------------------------------Finish Input Definition ------------------------------------------------
% Parameter setting
% 1) Functional parameters
FlagRemoveVerticalSupport = 1; % Enable Removing Vertical Support Depth at the second inference (risky if occluded)
FlagEnableVarMap = 0; % Disabled 
gravity =true; % if true, apply the HoriConf and VertConf linear scale weight
CoPST = true; % if true, apply the Straight line prior as the Co-Planar constrain
% ============= Magic Number =============
StickHori = 5;  %0.1; % sticking power in horizontal direction
StickVert = 10;     % sticking power in vertical direction
Center = 2; % Co-Planar weight at the Center of each superpixel
HoriConf = 1; % set the confidant of the learned depth at the middle in Horizontal direction of the image
VertConf = 0.01; % set the confidant of the learned depth at the top of the image
BandWith = 1; % Nov29 1 Nov30 0.1 check result change to 0.1 12/1 0.1 lost detail
ClosestDist = 1; % define the closest distance that the MRF allows
FarestDist = 1.5*median(depthMap(:)); % tried on university % nogood effect but keep it since usually it the rangelike this   % change to 1.5 for church	
% The hand-tuned 14 dimensional vector correspose to the weights of 14 multiple segmentation
if ~isempty(MultiScaleSupTable)
	MultiScaleFlag = true;	% multiple segmentaition hypothesis
   	WeiV = 2*ones(1,size(MultiScaleSupTable,2)-1);
else
   	MultiScaleFlag = false;
   	WeiV = 1;
end
WeiV(1,1:2:end) = 6; % emphasize the middle scale three times smaller than large scale
WeiV = WeiV./sum(WeiV);% normalize if pair of superpixels have same index in all the scale, their weight will be 10
ShiftStick = -.1;  % between -1 and 0, more means more smoothing.
ShiftCoP = -.5;  % between -1 and 0, more means more smoothing.
 % =========================================================================
% If you go up in the vertical direction in the image, the weights change in the vertical direction.
groundThreshold = cos([ zeros(1, Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)+10) ...
				linspace(0,15,ceil(Default.VertYNuDepth/2)-10)]*pi/180);
%  v1 15 v2 20 too big v3 20 to ensure non misclassified as ground.
%  verticalThreshold = cos(linspace(5,55,Default.VertYNuDepth)*pi/180); % give a vector of size 55 in top to down : 
verticalThreshold = cos([ 5*ones(1,Default.VertYNuDepth - ceil(Default.VertYNuDepth/2)) ...
				linspace(5,55,ceil(Default.VertYNuDepth/2))]*pi/180); 
% give a vector of size 55 in top to down : 
% 50 means suface norm away from y axis more than 50 degree
% =========================================================================
ceiling = 0*Default.VertYNuDepth; % set the position of the ceiling, related to No plane coming back constrain % changed for newchurch
% ^^^^^This number means the lowest row that planes may go back just like a ceiling
% ============= End of Magic Number ======

% 2) Opimization parameters
% ============== parameters for the decomposition problem
% Optimal parameters for current code:
% For one machine: Sedumi: (1,1)        Lpsolve:(3,1)
% Multiple machines: Sedumi: (4,1)      LPsolve:(3,1)
% lpsolve running time is 22 seconds for (4,2) arrangement; but numerical
% accuracy needs to be resolved first.
XNuDecompose = 1; % up to 3 is stable      
YNuDecompose = 1;
% ============ parameters for the decomposition problem
solverVerboseLevel = 0; % set to 1 if you need msg out when solving optimization problem
NoSecondStep = 0; % set to 1 if only want first level opt = no second step of vertical and horizontal objective

% 3) Debug and evaluation paramters
ExtractRelationInfo = 1; % set to 1 if need to storage the coplanar and connectivity weight to analyze

% 4) Rendering parameters
GridFlag = 0; % set to 1 if the wrl need grid of the triangle overlay

% ==========================End of parameter setting ====================================
inferenceTime = tic;
fprintf(['\n     : Building Matrices....       ']);

% ======= intermediant Data =====================================================================
% confidence of the supporting depth changes according to the row and column =========
mapVert = linspace(VertConf,1,Default.VertYNuDepth); % modeling the gravity prior:  the topper the row is the lower the confidence of supporting depths
mapHori = [linspace(HoriConf,1,round(Default.HoriXNuDepth/2)) fliplr(linspace(HoriConf,1,Default.HoriXNuDepth-round(Default.HoriXNuDepth/2)))];
% the more peripheral column the lower the confidence of supporting depths
% ====================================================================================
% assign the confidance of the supporting depths
if FlagEnableVarMap
	VarMap = zeros( size(VarMapRaw));
else
	VarMap = VarMapRaw;
end
CleanedDepthMap = depthMap;
CleanedDepthMap(depthMap>FarestDist) = NaN; % set the supporting depths ti NaN means that depth is not effective in OPT, so we simply ignore the depth which is too far
Posi3D = im_cr2w_cr(CleanedDepthMap,permute(RayOri,[2 3 1])); % given the depth and ray as input, calculate the 3-d coordinate at each point.

% Clean the Sup near sky
maskSky = Sup == 0;
maskSkyEroded = imerode(maskSky, strel('disk', 4) );
SupEpand = ExpandSup2Sky(Sup,maskSkyEroded); % extend the sky using the cloesest Sup index, which mean the some part of the sky will be include in the sup but their supporting depth will not be used.

NuSup = setdiff(unique(Sup)',0); % unique index of sup (not including sky index which is '0')
NuSupSize = size(NuSup,2); % number of uniquew sup

% Sup index and planeParameter index inverse map ======= useful tool======
% since sup index is not continuous, but the Parameter index over sup is continuous. Sup2Para is the maping stand for sup index to Parameter index
Sup2Para = sparse(1,max(Sup(:))); 
Sup2Para(NuSup) = 1:NuSupSize;
% ========================================================================

% =================please ignore here since StraightLineTable = [], NuLine = 0 ========
% constructiion of the Straight line prior matrix Will be add in the CoPlane matrix
NuLine = size(StraightLineTable,2);
CoPSTList = [];

% effectively not running the straight line constraint here.
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
% Please ignore ===============================================================

% ============end of generating intermediant Data ===============================================

% Generate the Matrix for MRF ==============================================================
PosiM = sparse(0,0); % supporting depth times ray terms
VarM = sparse(0,0); % confidence terms
RayAllM = sparse(0,0); % collection of every ray

% keep record of the center of the sup is lower then the ceiling
YPointer = []; % set to one if lower then the ceiling
YPosition = []; % keep a record of the row of the center of the sup, useful when enforcing vertical and horizontal constrain

for i = NuSup
    	mask = SupEpand ==i; % include the Ray that will be use to expand the NonSky
    	RayAllM = blkdiag( RayAllM, Ray(:,mask)'); % RayAllM will be putting on constrain of every ray that will be rendered, so should use SupEpand as mask

    	mask = Sup ==i; % Not include the Ray that will be use to expand the NonSky    
    	[yt x] = find(mask);
	
	% find the center point of the sup
    	CenterX = round(median(x));
    	CenterY = round(median(yt));

	% if CenterY is lower than ceiling, then set YPointer for that sup to 1 so that it will not come back like a ceiling
    	YPointer = [YPointer; CenterY >= ceiling]; 

	% the horizontal and vertical enforcing in the second step OPT will depends on the height of the sup, so keep record of CenterY
    	YPosition = [YPosition; CenterY];

	% Not building PosiM and VarM, have to get rid of NaN depths
    	mask(isnan(CleanedDepthMap)) = false;
    	SupNuPatch(i) = sum(mask(:));
    
	% find the center point of the sup
    	[yt x] = find(mask); % notice VarM depends on the position of mask if gravity == 1
  
  	if ~all(mask(:)==0)
    		if gravity
      			if any(CleanedDepthMap(mask) <=0)
         			CleanedDepthMap(mask)
      			end
      			PosiM = blkdiag(PosiM,Posi3D(:,mask)');%.*repmat( mapVert(yt)',[1 3]).*repmat( mapHori(x)',[1 3]));
      			VarM = [VarM; VarMap(mask).*(mapVert(yt)').*( mapHori(x)')];
   	 	else
      			PosiM = blkdiag(PosiM,Posi3D(:,mask)');
      			VarM = [VarM; VarMap(mask)];
    		end
  	else
     		PosiM = blkdiag(PosiM, Posi3D(:,mask)');
     		VarM = [VarM; VarMap(mask)];
  	end
end
% set 0 to -1 since this will make the sup higher than the ceiling enforce to come back, if you do not want this just keep 0 as 0
YPointer(YPointer==0) = -1;

% buliding CoPlane Matrix=========================================================================

% ===============please ignore here since CoPSTList = [], ===============
NuSTList = 0;
if CoPST
   	NuSTList = size(CoPSTList,1);
   	if ~isempty(CoPSTList)
      		[V H] = size(SupNeighborTable);
      		SupNeighborTable( CoPSTList(:,1)*V + CoPSTList(:,2)) = 1;
      	SupNeighborTable( CoPSTList(:,2)*V + CoPSTList(:,1)) = 1;
   	end
end
% ================please ignore==============================================

CoPM1 = sparse(0,3*NuSupSize);
CoPM2 = sparse(0,3*NuSupSize);
CoPEstDepth = sparse(0,0);
CoPNorM = [];
WeiCoP = [];
if ExtractRelationInfo == 1
	% keeps the Wei of the relational Coplanar term here
	WeiM = sparse(max(NuSup),max(NuSup));
end

for i = NuSup
	% pick the i's neightbot using SupNeighborTable
    	Neighbor = find( SupNeighborTable(i,:) ~=0);
    	Neighbor = Neighbor( Neighbor> i);
	% setup the relation between sup_i and sup_j since they are neighbors
    	for j = Neighbor
        	mask = Sup == i;
        	SizeMaskAll = sum(mask(:));
        	[y x] = find(mask);
		% Coplanar term only use the center ray of each sup to enfore the cost
        	CenterX = round(median(x));
        	CenterY = round(median(y));
        	y = find(mask(:,CenterX));
        	if ~isempty(y)
           		CenterY = round(median(y));
        	end
        	temp1 = sparse(1, 3*NuSupSize);
        	temp2 = sparse(1, 3*NuSupSize);
        	temp1(:,(Sup2Para( i)*3-2): Sup2Para( i)*3) = ...
                      RayOri(:,CenterY,CenterX)';
        	temp2(:,(Sup2Para( j)*3-2): Sup2Para( j)*3) = ...
                      RayOri(:,CenterY,CenterX)';

		% assign wei for each pairs of neighbors
           	if MultiScaleFlag 
              		vector = (MultiScaleSupTable(Sup2Para( i),2:end) == MultiScaleSupTable(Sup2Para( j),2:end));
              		expV = exp(-10*(WeiV*vector' + ShiftCoP) );
              		wei = 1/(1+expV);
           	else
              		wei = 1;
           	end
	
	   	if ExtractRelationInfo == 1; % keep record
			WeiM(i,j) = wei;
	   	end

        	oneRay1 = temp1*wei;
        	oneRay2 = temp2*wei;
    
        	tempWeiCoP = [SizeMaskAll];
        	CoPEstDepth = [CoPEstDepth; max(median(CleanedDepthMap(mask)),ClosestDist)];
    
        	mask = Sup == j;
        	SizeMaskAll = sum(mask(:));
        	[y x] = find(mask);
		
		% Coplanar term only use the center ray of each sup to enfore the cost
        	CenterX = round(median(x));
        	CenterY = round(median(y));
        	y = find(mask(:,CenterX));
        	if ~isempty(y)
           		CenterY = round(median(y));
       		end

        	temp1 = sparse(1, 3*NuSupSize);
        	temp2 = sparse(1, 3*NuSupSize);
        	temp1(:,(Sup2Para( i)*3-2): Sup2Para( i)*3) = ... 
                      RayOri(:,CenterY,CenterX)';
        	temp2(:,(Sup2Para( j)*3-2): Sup2Para( j)*3) = ...
                      RayOri(:,CenterY,CenterX)';
    
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

% Ashutosh added to reduce linear dependancy of the objective =======
SupPixelNeighborList = sparse( max(Sup(:)), max(Sup(:)) );
SupPixelParsedList = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded1 = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded2 = sparse( max(Sup(:)), max(Sup(:)) );
addedIndexList = [ ];
% ===================================================================

% locate sup boundary at horizontal and vertical dircetion only for stitching terms
BounaryPHori = conv2(Sup,[1 -1],'same') ~=0;
BounaryPHori(:,end) = 0;
BounaryPVert = conv2(Sup,[1; -1],'same') ~=0;
BounaryPVert(end,:) = 0;

%    boundariesHoriIndex = find(BounaryPHori==1)';
% pre-select the boundary in order with the NuSup order
for l = NuSup
    	mask = Sup == l;
    	boundariesHoriIndex = find(BounaryPHori==1 & mask)';
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
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    this is supposed to be where the y and nu information are computed,
%%%%    keep an eye on this.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after analysis finally add to HoriStickM_j
WeightHoriNeighborStitch = [ ];
for i = addedIndexList
    	j = i+Default.VertYNuDepth;
    	WeightHoriNeighborStitch = [WeightHoriNeighborStitch;  SupPixelParsedList(Sup(i),Sup(j)) / ...
                                    SupPixelNeighborList(Sup(i),Sup(j)) ];

    	Target(1) = Sup2Para(Sup(i));
    	Target(2) = Sup2Para(Sup(j));
    	rayBoundary(:,1) =  RayOri(:,i);
    	rayBoundary(:,2) =  RayOri(:,i);
    	if MultiScaleFlag
          	vector = (MultiScaleSupTable(Sup2Para(Sup(i)),2:end) == MultiScaleSupTable(Sup2Para(Sup(j)),2:end));
          	expV = exp(-10*(WeiV*vector' + ShiftStick) );
       		betaTemp = StickHori*(0.5+1/(1+expV));
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    end of y and nu computation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ==============================================

% ======== finding the unucessary stiching points in Vertical direction ====
VertStickM_i = sparse(0,3*NuSupSize);
VertStickM_j = sparse(0,3*NuSupSize);
VertStickPointInd = [];
EstDepVertStick = [];

MAX_POINTS_STITCH_VERT = 4;	%3
DIST_STICHING_THRESHOLD_VERT = 0.1;	%0.3 
DIST_STICHING_THRESHOLD_VERT_ONLYCOL = -0.5;    % effectively not used, ideally should be 0.5; i.e., the point should be farther in col direction because that is the direction of the edge.

% Ashutosh added to reduce linear dependancy of the objective =======
SupPixelNeighborList = sparse( max(Sup(:)), max(Sup(:)) );
SupPixelParsedList = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded1 = sparse( max(Sup(:)), max(Sup(:)) );
recordAdded2 = sparse( max(Sup(:)), max(Sup(:)) );
addedIndexList = [ ];
% ===============================

% pre-select the boundary in order with the NuSup order
for l = NuSup
	mask = Sup == l;
    	for i = find(BounaryPVert==1 & mask)'
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
end

% after analysis finally add to VertStickM_j
WeightVertNeighborStitch = [ ];
for i = addedIndexList
    	j = i+1;
     	WeightVertNeighborStitch = [WeightVertNeighborStitch;  SupPixelParsedList(Sup(i),Sup(j)) / ...
                                    SupPixelNeighborList(Sup(i),Sup(j)) ];
    	Target(1) = Sup2Para(Sup(i));
    	Target(2) = Sup2Para(Sup(j));
   	rayBoundary(:,1) =  RayOri(:,i);
    	rayBoundary(:,2) =  RayOri(:,i);
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
% Start Decompose the image align with superpixels ==================================================================================
% define the decomposition in both X and Y direction
 
TotalRectX = 2*XNuDecompose-1;
TotalRectY= 2*YNuDecompose-1;
PlanePara = NaN*ones(3*NuSupSize,1); % setup the lookup table for the solved plane parameter

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
    
		if any(SubSup <=0)
       			SubSup(SubSup<=0)
    		end
    		BoundarySup = find(sum(SupNeighborTable(SubSup,:), 1) ~=0);
    		BoundarySup = unique(setdiff(BoundarySup,[0 SubSup] ));
    
		% chech if BoundarySup non-NaN in PlanePara
    		checkNoNNaN = ~isnan(PlanePara(Sup2Para(BoundarySup)*3));
    		BoundarySup = BoundarySup(checkNoNNaN);
    		TotalSup = sort([SubSup BoundarySup]);

		% define the pointer to extract the data for this specific decomposition
    		SubSupPtr = [ Sup2Para(SubSup)*3-2;...
                    	      Sup2Para(SubSup)*3-1;...
                    	      Sup2Para(SubSup)*3];
    		SubSupPtr = SubSupPtr(:); 
	    	BoundarySupPtr = [ Sup2Para(BoundarySup)*3-2;...
			           Sup2Para(BoundarySup)*3-1;...
			           Sup2Para(BoundarySup)*3];
    		BoundarySupPtr = BoundarySupPtr(:);
    		NuSubSupSize = size(SubSup,2);
    
    		% simply extract  NewRayAllM NewPosiM NewCoPM =========================================
	        % NewHoriStickM NewVertStickM for the specific decomposition 
    		NewRayAllM = RayAllM(:,SubSupPtr);
    		tar = sum(NewRayAllM ~= 0,2) == 3;
    		NewRayAllM = NewRayAllM(tar,:);
    		NewPosiM = PosiM(:,SubSupPtr);
    		tar = sum(NewPosiM ~= 0,2) == 3;
    		NewPosiM = NewPosiM(tar,:);
    		NewVarM = VarM(tar);
    
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
   
		WeightsSelfTerm = 1 ./ exp(abs(NewVarM)/BandWith);
    		% ==========================end of extraction ====================

		fprintf(['     ' num2str( toc(inferenceTime) ) '\n     : In 1st level Optimization, using new solver.' ...
			'(' num2str(k+1) '/' num2str((TotalRectX-1)+1) ',' num2str(l+1) '/' num2str((TotalRectY-1)+1) ')']);

		global A b S inq

		A = [ sparse(1:length(WeightsSelfTerm),1:length(WeightsSelfTerm),WeightsSelfTerm ) * NewPosiM;...
			sparse(1:length(NewCoPEstDepth),1:length(NewCoPEstDepth),NewCoPEstDepth*Center ) * NewCoPM;...
			sparse(1:length(NewEstDepHoriStick), 1:length(NewEstDepHoriStick), NewEstDepHoriStick.*NewWeightHoriNeighborStitch) * NewHoriStickM;...
			sparse(1:length(NewEstDepVertStick), 1:length(NewEstDepVertStick), NewEstDepVertStick.*NewWeightVertNeighborStitch) * NewVertStickM;...
			];

		b = [ ones(size(NewPosiM,1),1) .* WeightsSelfTerm;...
			-NewCoPMBound.*NewCoPEstDepth*Center;...
			-NewHoriStickMBound.*NewEstDepHoriStick.*NewWeightHoriNeighborStitch;...
			-NewVertStickMBound.*NewEstDepVertStick.*NewWeightVertNeighborStitch];
 
		temp = zeros(1, NuSubSupSize*3);
		temp(3*(1:NuSubSupSize)-1) = YPointer(Sup2Para(SubSup));
		temp = sparse(1:length(temp), 1:length(temp), temp);
		temp( sum(temp,2) ==0,:) = [];
		S = [temp;...
			NewRayAllM;...
			-NewRayAllM];
		inq = [ sparse(size(temp,1), 1);...
			- 1/ClosestDist*ones(size(NewRayAllM,1),1);...
			1/FarestDist*ones(size(NewRayAllM,1),1)];
		Para.ClosestDist = ClosestDist;
		Para.FarestDist = FarestDist;
		Para.ptry = [];
		Para.ptrz = [];
		Para.Dist_Start = [];

		% solve the OPT problem using the new solver
		[x_ashIterator, alfa, status, history, T_nt_hist] = ...
			SigmoidLogBarrierSolver( Para, [], [], [], '', [], [], solverVerboseLevel);
		
		% check if the constrian still satisfied
		if any(S*x_ashIterator+inq > 0 )
			disp('Inequality not satisfied');
			max( S*x_ashIterator+inq)
		elseif status == 2
			fprintf([' Success with alfa=' num2str(alfa)]); 
		end

		% assign the solution of specific decomposition to the corresponding entries of the whole problems
    		PlanePara(SubSupPtr) = x_ashIterator;
    	end
end

% build the whole image
PlanePara = reshape(PlanePara,3,[]);
% porject the ray on planes to generate the ProjDepth
FitDepthPPCP = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
FitDepthPPCP(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded ))).*Ray(:,~maskSkyEroded ),1))';
FitDepthPPCP = reshape(FitDepthPPCP,Default.VertYNuDepth,[]);
[Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));

if NoSecondStep % if no second step
	Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
        Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
        RR =permute(Ray,[2 3 1]);
        temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
        PositionTex = permute(temp./repmat(cat(3,Default.a_default,Default.b_default),[Default.VertYNuDepth Default.HoriXNuDepth 1])+repmat(cat(3,Default.Ox_default,Default.Oy_default),[Default.VertYNuDepth Default.HoriXNuDepth 1]),[3 1 2]);
        PositionTex = permute(PositionTex,[2 3 1])
	disp('First level Wrl');

	% write wrl file
        WrlFacestHroiReduce(Position3DFitedPPCP,PositionTex,SupOri, [ Default.filename{1} '1st'],[ Default.filename{1} '1st'], ...
                            Default.OutPutFolder, GridFlag, 0);
        system(['gzip -9 -c ' Default.OutPutFolder Default.filename{1} '1st.wrl > ' ...
               Default.OutPutFolder Default.filename{1} '1st.wrl.gz']);
        copyfile([ Default.OutPutFolder Default.filename{1} '1st.wrl.gz '], ...
               [ Default.OutPutFolder Default.filename{1} '1st.wrl'],'f');
        delete([Default.OutPutFolder Default.filename{1} '1st.wrl.gz']);
end
%==================Finished for one step MRF==================================================================================================

if NoSecondStep
	return;
end

% ====================================following are 2nd step MRF to give more visually pleasing result=======================================
% generating new PosiMPPCP using the new position
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
   
% forming new supporting matrix using new depth and get rid of the support of the vertical plane
for i = NuSup
	mask = Sup == i;
      	if any(verticalParaInd == Sup2Para(i)) & FlagRemoveVerticalSupport % Remove Vertical Depths Supports
         	mask = logical(zeros(size(Sup)));
      	end
        PosiMPPCP = blkdiag(PosiMPPCP, Position3DFitedPPCP(:,mask)');
        VarM2 = [VarM2; VarMap(mask)];
        if length( find(isnan(PosiMPPCP)) )
        	disp('PosiMPPCP is NaN');
        end
end

% Start Decompose image =========================
TotalRectX = 2*XNuDecompose-1;
TotalRectY = 2*YNuDecompose-1;
PlanePara = NaN*ones(3*NuSupSize,1); % setup the lookuptable for the solved plane parameter

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
      		SubSup = sort(setdiff( unique( reshape( Sup(RangeY,RangeX),1,[])),0));
      		BoundarySup = [];
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
      		BoundarySupPtr =BoundarySupPtr(:);
      		NuSubSupSize = size(SubSup,2);
      		SubSup2Para = sparse(1,max(SubSup));
      		SubSup2Para(SubSup) = 1:NuSubSupSize;
    
      		% clearn RayAllM PosiM CoPM1 HoriStickM_i VertStickM_i
      		NewRayAllM = RayAllM(:,SubSupPtr);
      		tar = sum(NewRayAllM ~= 0,2) ==3;
      		NewRayAllM = NewRayAllM(tar,:);

      		NewPosiMPPCP = PosiMPPCP(:,SubSupPtr);
      		tar = sum(NewPosiMPPCP ~= 0,2) ==3;
     	 	NewPosiMPPCP = NewPosiMPPCP(tar,:);
      		NewVarM = VarM2(tar);

      		NewCoPM = CoPM1(:,SubSupPtr) - CoPM2(:,SubSupPtr);
      		NewCoPMBound = CoPM1(:,BoundarySupPtr) - CoPM2(:,BoundarySupPtr);
      		tar = sum( NewCoPM ~= 0,2) + sum( NewCoPMBound ~= 0,2) ==6;
      		NewCoPM = NewCoPM(tar,:);
      		NewCoPMBound = NewCoPMBound*PlanePara(BoundarySupPtr); % column vertor
      		NewCoPMBound = NewCoPMBound(tar);
      		NewCoPEstDepth = CoPEstDepth(tar);   

      		NewHoriStickM = HoriStickM_i(:,SubSupPtr)-HoriStickM_j(:,SubSupPtr);
      		NewHoriStickMBound = HoriStickM_i(:,BoundarySupPtr)-HoriStickM_j(:,BoundarySupPtr);
      		tar = sum(NewHoriStickM ~= 0,2) + sum( NewHoriStickMBound ~= 0,2)==6;
      		NewHoriStickM = NewHoriStickM(tar,:);
      		NewHoriStickMBound = NewHoriStickMBound*PlanePara(BoundarySupPtr); % column vertor
      		NewHoriStickMBound = NewHoriStickMBound(tar);
      		NewEstDepHoriStick = EstDepHoriStick(tar);
         	NewWeightHoriNeighborStitch = WeightHoriNeighborStitch(tar);

      		NewVertStickM = VertStickM_i(:,SubSupPtr)-VertStickM_j(:,SubSupPtr);
      		NewVertStickMBound = VertStickM_i(:, BoundarySupPtr)-VertStickM_j(:,BoundarySupPtr);
      		tar = sum(NewVertStickM ~= 0,2) + sum(NewVertStickMBound ~= 0,2)==6;
      		NewVertStickM = NewVertStickM(tar,:);
      		NewVertStickMBound = NewVertStickMBound*PlanePara(BoundarySupPtr); % column vertor
      		NewVertStickMBound = NewVertStickMBound(tar);
      		NewEstDepVertStick = EstDepVertStick(tar);
         	NewWeightVertNeighborStitch = WeightVertNeighborStitch(tar);

		%    try reduce the vertical constrain
     		NonVertPtr = setdiff( 1:(3*NuSubSupSize), SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))*3-1 );
     		YNoComingBack = YPointer(Sup2Para(SubSup));
     		YNoComingBack(SubSup2Para( NuSup( (intersect( indexVertical,SubSupPtr ) +1)/3))) = [];
     		YCompMask = zeros(1,3*NuSubSupSize);
     		YCompMask(3*(1:NuSubSupSize)-1) = 1;
     		YCompMask = YCompMask(NonVertPtr); 
     		XCompMask = zeros(1,3*NuSubSupSize);
     		XCompMask( SubSup2Para( NuSup( (intersect( indexGroundX,SubSupPtr ) +2)/3))*3-2 ) = 1;
     		XCompMask = XCompMask(NonVertPtr);
     		ZCompMask = zeros(1,3*NuSubSupSize);
     		ZCompMask( SubSup2Para( NuSup( (intersect( indexGroundZ,SubSupPtr ) )/3))*3 ) = 1;
     		ZCompMask = ZCompMask(NonVertPtr);
    	 	GroundMask = intersect( find(groundPara), ( SubSupPtr( 3:3:( size(SubSupPtr,1)) )/3 ) ) ;

		% version written by Ashutosh

		tempGroundX = sparse(1:length(XCompMask), 1:length(XCompMask), XCompMask);
		tempGroundX( logical(XCompMask) ) = tempGroundX( logical(XCompMask) )./normPara( GroundMask );
		tempGroundX( sum(tempGroundX,2) == 0,:) = [];
		tempGroundZ = sparse(1:length(ZCompMask), 1:length(ZCompMask), ZCompMask );
		tempGroundZ( logical(ZCompMask) ) = tempGroundZ( logical(ZCompMask) )./normPara( GroundMask );
		tempGroundZ( sum(tempGroundZ,2) == 0,:)= [];

		A = [	sparse(1:length(NewVarM),1:length(NewVarM),1./exp(abs(NewVarM)/BandWith)) * NewPosiMPPCP(:, NonVertPtr);...
			sparse(1:length(NewCoPEstDepth), 1:length(NewCoPEstDepth), NewCoPEstDepth * Center) * NewCoPM(:, NonVertPtr);...
			sparse(1:length(NewEstDepHoriStick), 1:length(NewEstDepHoriStick), NewEstDepHoriStick.*NewWeightHoriNeighborStitch) * NewHoriStickM(:, NonVertPtr);...
			sparse(1:length(NewEstDepVertStick), 1:length(NewEstDepVertStick), NewEstDepVertStick.*NewWeightVertNeighborStitch) * NewVertStickM(:, NonVertPtr);...
			tempGroundX;...
			tempGroundZ...
			];
		%        +10*norm( ( Para( logical(XCompMask) ))./...
		%                            normPara( GroundMask )', 1)...
		%                 +10*norm( ( Para( logical(ZCompMask)))./...
		%                            normPara( GroundMask )', 1) ...
 
		b = [	1./exp(abs(NewVarM)/BandWith); ...
			-Center*NewCoPMBound.*NewCoPEstDepth; ...
			-NewHoriStickMBound.*NewEstDepHoriStick.*NewWeightHoriNeighborStitch;...
			-NewVertStickMBound.*NewEstDepVertStick.*NewWeightVertNeighborStitch;...
			sparse(size(tempGroundX,1),1);...
			sparse(size(tempGroundZ,1),1)...
			];

		temp = YCompMask;
		temp(logical(YCompMask)) = YNoComingBack;
		temp = sparse(1:length(temp), 1:length(temp), temp);
		temp( sum(temp,2) ==0,:) = [];
		S = [   temp;...
			NewRayAllM(:,NonVertPtr);...
			-NewRayAllM(:,NonVertPtr);...
			];
		inq = [ sparse(size(temp,1), 1);...
			- 1/ClosestDist*ones(size(NewRayAllM,1),1);...
			1/FarestDist*ones(size(NewRayAllM,1),1);...
			];

		Para.ClosestDist = ClosestDist;
		Para.FarestDist = FarestDist;

		% build up ptry and ptrz adapt fot NonVertPtr
		Para.ptry = zeros(size(NewRayAllM,2),1);
		Para.ptry(2:3:size(NewRayAllM,2)) = 1;
		Para.ptry = logical(Para.ptry(NonVertPtr));
		Para.ptrz = zeros(size(NewRayAllM,2),1);
		Para.ptrz(3:3:size(NewRayAllM,2)) = 1;
		Para.ptrz = logical(Para.ptrz(NonVertPtr));
		Para.Dist_Start = size(temp,1)+1;
		[x_ashIterator, alfa, status] = SigmoidLogBarrierSolver(Para, [], [], [], '', [], [], solverVerboseLevel);
		if any(S*x_ashIterator+inq > 0 )
			disp('Inequality not satisfied');
			max( S*x_ashIterator+inq)
		elseif status==2
			fprintf([' Success with alfa=' num2str(alfa)]); 
		end

		Para = x_ashIterator;

		tempPara = zeros(3*NuSubSupSize,1);
		tempPara(NonVertPtr) = Para;
		PlanePara(SubSupPtr) = tempPara;
	end
end

% porject the ray on planes to generate the FitDepth
PlanePara = reshape(PlanePara,3,[]);
FitDepth = FarestDist*ones(1,Default.VertYNuDepth*Default.HoriXNuDepth);
FitDepth(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded))).*Ray(:,~maskSkyEroded),1))';
FitDepth = reshape(FitDepth,Default.VertYNuDepth,[]); 
% ==========Storage ==============
if Default.Flag.AfterInferenceStorage
	save([ Default.ScratchFolder '/' strrep( Default.filename{1},'.jpg','') '_AInfnew.mat' ], 'FitDepth', 'depthMap', ...
            'Sup', 'SupOri', 'RayOri','Ray','SupNeighborTable','maskSky','maskG','MultiScaleSupTable','WeiM','VarMapRaw');
end
% ===============================
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

% write wrl file
fprintf(['     ' num2str( toc(inferenceTime) ) '\n     : Writing WRL.']);
WrlFacestHroiReduce(Position3DFited, PositionTex, SupOri, Default.filename{1}, Default.filename{1}, ...
Default.OutPutFolder, GridFlag, 0);

system(['gzip -9 -c ' Default.OutPutFolder Default.filename{1} '.wrl > ' ...
	Default.OutPutFolder Default.filename{1} '.wrl.gz']);
copyfile([Default.OutPutFolder Default.filename{1} '.wrl.gz'], ...
	[Default.OutPutFolder Default.filename{1} '.wrl'],'f');
delete([Default.OutPutFolder Default.filename{1} '.wrl.gz']);

return;
