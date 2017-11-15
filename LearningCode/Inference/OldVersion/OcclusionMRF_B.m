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
function []=OcclusionMRF(k);

% not working perfectly. More search needed.

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

% This function run a boundary MRF using the laser data to estimate the occlusion boundary
tic
% set Parameters
WSLInitialWei = 0.5;
WCornerInitialWei = 0.3;
SLExcludeHoriWei = 5; % Hori have a bigger resolusion so have a bigger Exlude region
SLExcludeVertWei = 10;
BpWidthV = ceil(0.01*VertYNuDepth); % assume uniform distibution of the shift of estimated Bp with Width = BpWidthV (Precentage in Vert diection)
BpWidthH = ceil(0.1*HoriXNuDepth); % BpWidthH (Precentage in Vert diection)

% prepare the data
%1) depth ready
depthfile = strrep(filename{k},'img','depth_sph_corr'); % the depth filename
load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
LaserDepth = Position3DGrid(:,:,4);
% 2) ProjXY ready
PicsinfoName = strrep(filename{k},'img','picsinfo');
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
RayProjY = repmat((1:VertYNuDepth)',[1 HoriXNuDepth]);
RayProjX = repmat((1:HoriXNuDepth),[VertYNuDepth 1]);
RayPorjectImgMapYImCo = ((VertYNuDepth+1-RayProjY)-0.5)/VertYNuDepth - Oy;
RayPorjectImgMapXImCo = (RayProjX-0.5)/HoriXNuDepth - Ox;
% 3) Ray ready
Ray = RayImPosition(RayPorjectImgMapYImCo,RayPorjectImgMapXImCo,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
Ray = permute(Ray,[3 1 2]);

% Vertex X Y position
VertexY = repmat((1:(VertYNuDepth+1))',[1 HoriXNuDepth+1]);
VertexX = repmat((1:(HoriXNuDepth+1)),[VertYNuDepth+1 1]);
 % change them into image coordinate
Vertex = Matrix2ImgCo(HoriXNuDepth+1, VertYNuDepth+1, [VertexX(:) VertexY(:)]);
VertexY = reshape(Vertex(:,2), VertYNuDepth+1, []);
VertexX = reshape(Vertex(:,1), VertYNuDepth+1, []);

% 4) Position 3D ready
%Posi3D = im_cr2w_cr(LaserDepth,permute(Ray,[2 3 1]));

% 5) Sup Ready
load([ScratchDataFolder '/data/CleanSup/CleanSup' num2str(k) '.mat']);
load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
LaserDepth = Position3DGrid(:,:,4);
[SegVertYSize, SegHoriXSize] = size(MedSup);
MedSup = double(MedSup);
Sup = double(Sup);
MaxSup = max( Sup(:));
MaskSky = Sup ==0;

% 6) Img Ready
Img = imresize(imread([GeneralDataFolder '/' ImgFolder '/' filename{k} ],'jpg'), [SegVertYSize SegHoriXSize]);

% 7) SupBounday with HashIndex Ready
BoundaryPVert = conv2(Sup,[MaxSup 1],'valid').*( conv2(Sup,[1 -1],'valid') >0)...
                .* ~MaskSky(:,1:(end-1)) .* ~MaskSky(:,2:end); % two step build up hash index 1) Left > Righ index
BoundaryPVert = BoundaryPVert + conv2(Sup,[1 MaxSup],'valid').* (conv2(Sup,[-1 1],'valid') >0)...
                .* ~MaskSky(:,1:(end-1)) .* ~MaskSky(:,2:end); % 2) Left < Righ index
BoundaryPHori = conv2(Sup,[MaxSup; 1],'valid').* (conv2(Sup,[1; -1],'valid') >0)...
                .* ~MaskSky(1:(end-1),:) .* ~MaskSky(2:end,:); % two step build up hash index 1) Top > bottom index
BoundaryPHori = BoundaryPHori + conv2(Sup,[1; MaxSup],'valid').*( conv2(Sup,[-1; 1],'valid') >0)...
                .* ~MaskSky(1:(end-1),:) .* ~MaskSky(2:end,:); % 2) Top < Bottom index
ClosestNHashList = setdiff(unique([BoundaryPHori(:); BoundaryPVert(:)]),0);
NuNei = size(ClosestNHashList,1);
MaxHash = max(ClosestNHashList(:));
Hash2Ind = sparse(1,MaxHash);
Hash2Ind(ClosestNHashList) = 1:NuNei;

% 8) Straight Line detection in Multi Scale
[seglist]=edgeSegDetection(Img,k,0);

figure(100); subplot(2,2,1);
ImgMapSup = Img;
ImgMapSup(:,:,1) = 255/max(Sup(:))*imresize(Sup, [ SegVertYSize SegHoriXSize]);
image(ImgMapSup);
drawseg(seglist,100);
PlotGridBoundary( (BoundaryPHori ~=0), (BoundaryPVert ~=0), VertexX, VertexY, [ SegVertYSize SegHoriXSize], 100, 'y');
% maping Straight line to BoundaryLineCross  matrix
[BoundaryLineCrossHori BoundaryLineCrossVert] = ...
    ST2BoundaryLineCross(seglist, BoundaryPHori, BoundaryPVert); % now seglist change to size of VertYNuDepth HoriXNuDepth
    % same size of BoundaryPHori with 

% Show result of all possible Bunday and the Straighe line detected
figure(100); subplot(2,2,1);
%PlotGridBoundary( ones(size( BoundaryLineCrossHori)),ones(size(
%BoundaryLineCrossVert)), VertexX, VertexY, [ SegVertYSize SegHoriXSize],
%100,'y'); % plot all the boundary
PlotGridBoundary( BoundaryLineCrossHori, BoundaryLineCrossVert, VertexX, VertexY, [ SegVertYSize SegHoriXSize], 100, 'g');

% Constructing W matrix
WL = sparse(NuNei,NuNei);
WI = sparse(NuNei,NuNei);
WIL = sparse(NuNei,NuNei);
% 1) inital preference
ii = 1:(size(BoundaryPVert,1)-1);
jj = 1:(size(BoundaryPVert,2));
mask = zeros(size(BoundaryPVert));
mask(ii,jj) = 1;
IndVert = find(mask);
mask = zeros(size(BoundaryPHori));
mask(ii,jj) = 1;
IndHori = find(mask);
ProximitySLHast = sort([[BoundaryPVert(IndVert) BoundaryPVert(IndVert+1)]; [BoundaryPHori(IndHori) BoundaryPHori(IndHori+size(BoundaryPHori,1))]],2);
ProximitySLHast( (ProximitySLHast(:,1) == 0 | ProximitySLHast(:,2) == 0),:) = [];
ProximityCornerHast = sort([[BoundaryPVert(IndVert) BoundaryPHori(IndHori)];...
                            [BoundaryPVert(IndVert) BoundaryPHori(IndHori+size(BoundaryPHori,1))];...
                            [BoundaryPVert(IndVert+1) BoundaryPHori(IndHori)];...
                            [BoundaryPVert(IndVert+1) BoundaryPHori(IndHori+size(BoundaryPHori,1))]],2);
ProximityCornerHast( (ProximityCornerHast(:,1) == 0 | ProximityCornerHast(:,2) == 0),:) = [];
IndProximitySLHast = sub2ind([NuNei NuNei],Hash2Ind(ProximitySLHast(:,1)),Hash2Ind(ProximitySLHast(:,2)));
IndProximityCornerHast = sub2ind([NuNei NuNei], Hash2Ind(ProximityCornerHast(:,1)), Hash2Ind(ProximityCornerHast(:,2)));
WI( IndProximitySLHast) = -WSLInitialWei;
WI( IndProximityCornerHast) = -WCornerInitialWei;
WIL( IndProximitySLHast) = -WSLInitialWei;
WIL( IndProximityCornerHast) = -WCornerInitialWei;

% 2) Straight Line link preference
NuSL = size(seglist,1);
for l = 1:NuSL
    IndLineCrossHori = find(BoundaryLineCrossHori == l);
    IndLineCrossVert = find(BoundaryLineCrossVert == l);
    % decide to shift hori or vert
    if any(IndLineCrossVert == IndLineCrossVert+size(BoundaryLineCrossVert,1))
       shiftHori = 1;
       shiftVert = 1;
    else
       shiftHori = size(BoundaryPHori,1);
       shiftVert = size(BoundaryPVert,1);
    end
    SLHashVert = BoundaryPVert(IndLineCrossVert);
    SLHashVert( SLHashVert == 0) = [];
    SLHashHori = BoundaryPHori(IndLineCrossHori);
    SLHashHori( SLHashHori == 0) = [];
%     too strong preference for constructing straight line
%     across long dist
%     [y x] = meshgrid([SLHashVert; SLHashHori],[SLHashVert;SLHashHori]); 
%     check = y == x;
%     y( check) =[];
%     x( check) =[];
%     PairHash = unique(sort([y(:) x(:)],2),'rows');
    SLHash = [SLHashVert; SLHashHori];
    PairHash = unique(sort( [SLHash(1:(end-1)) SLHash(2:(end)) ], 2),'rows');
    if isempty(PairHash)
       continue; 
    end
    PairHash( PairHash(:,1) == PairHash(:,2),:) = [];
    if isempty(PairHash)
       continue; 
    end
    SLInd = sub2ind([NuNei NuNei], Hash2Ind(PairHash(:,1)), Hash2Ind(PairHash(:,2)));
    WL( SLInd) = -1; % -1 if prefer same label
    WIL( SLInd) = -1; % -1 if prefer same label
%    ExludePairHash1 = [ [reshape(SLHashVert(:,ones(1,SLExcludeVertWei)),[],1); reshape(SLHashHori(:,ones(1,SLExcludeHoriWei)),[],1)] ...
%                      [BoundaryPVert( reshape( max( min( repmat(IndLineCrossVert,[1 SLExcludeVertWei]) + ...
%                       repmat(shiftVert*(1:SLExcludeVertWei),[size(IndLineCrossVert,1) 1]), prod(size(BoundaryPVert))), 1), [],1) ) ;...
%                      BoundaryPHori( reshape( max( min( repmat(IndLineCrossHori,[1 SLExcludeHoriWei]) + ...
%                       repmat(shiftHori*(1:SLExcludeHoriWei),[size(IndLineCrossHori,1) 1]), prod(size(BoundaryPHori))), 1), [],1) ) ]...
%                      ];
%    ExludePairHash1( (ExludePairHash1(:,1) == 0 | ExludePairHash1(:,2) == 0),:) = [];                      
%    ExludePairHash2 = [[reshape(SLHashVert(:,ones(1,SLExcludeVertWei)),[],1); reshape(SLHashHori(:,ones(1,SLExcludeHoriWei)),[],1)]...
%                      [BoundaryPVert( reshape( max( min( repmat(IndLineCrossVert,[1 SLExcludeVertWei]) + ...
%                       repmat(-shiftVert*(1:SLExcludeVertWei),[size(IndLineCrossVert,1) 1]), prod(size(BoundaryPVert))), 1), [],1) ) ;...
%                      BoundaryPHori( reshape( max( min( repmat(IndLineCrossHori,[1 SLExcludeHoriWei]) + ...
%                       repmat(-shiftHori*(1:SLExcludeHoriWei),[size(IndLineCrossHori,1) 1]), prod(size(BoundaryPHori))), 1), [],1) ) ]...
%                      ];
%    ExludePairHash2( (ExludePairHash2(:,1) == 0 | ExludePairHash2(:,2) == 0),:) = [];
%    ExludePairHash = unique(sort([ExludePairHash1; ExludePairHash2],2), 'rows');
%    SLExludeInd = sub2ind([NuNei NuNei], Hash2Ind(ExludePairHash(:,1)), Hash2Ind(ExludePairHash(:,2)));
%    DonNotExcludeInd = W(SLExludeInd) == 1;
%    W( SLExludeInd(~DonNotExcludeInd)) = 1;
end

% 3) line complete and corner complete preference
% still don't know

% 4) parallel exlude 
NewHash2Ind =Hash2Ind;
NewHash2Ind(end+1) = NuNei+1;
NewBoundaryPVert = BoundaryPVert;
NewBoundaryPVert( NewBoundaryPVert==0) = size(NewHash2Ind,2);
SV = spalloc((size(NewBoundaryPVert,2)-(SLExcludeVertWei-1))*size(NewBoundaryPVert,1),NuNei+1,...
            (size(NewBoundaryPVert,2)-(SLExcludeVertWei-1))*size(NewBoundaryPVert,1)*SLExcludeVertWei);
InitialRow = 1;
for i = 1:SLExcludeVertWei
    ResV = rem(size(BoundaryPVert,2)-i+1,SLExcludeVertWei);
    XI = NewHash2Ind(reshape( NewBoundaryPVert(:,i:(end-ResV) )', SLExcludeVertWei, [] ));
    NuRow = size(XI,2);
    YI = repmat(InitialRow:(InitialRow+NuRow-1), [SLExcludeVertWei 1]);
    InitialRow = InitialRow+NuRow;
    SV(sub2ind(size(SV),YI(:),XI(:))) = 1;
end   
SV(:,end) = [];
mask = sum(SV,2) ==0;
SV(mask,:) = [];

NewBoundaryPHori = BoundaryPHori;
NewBoundaryPHori( NewBoundaryPHori==0) = size(NewHash2Ind,2);
SH = spalloc((size(NewBoundaryPHori,1)-(SLExcludeHoriWei-1))*size(NewBoundaryPHori,2),NuNei+1,...
            (size(NewBoundaryPHori,1)-(SLExcludeHoriWei-1))*size(NewBoundaryPHori,2)*SLExcludeHoriWei);
InitialRow = 1;
for i = 1:SLExcludeHoriWei
    ResH = rem(size(BoundaryPHori,1)-i+1,SLExcludeHoriWei);
    XI = NewHash2Ind(reshape( NewBoundaryPHori(i:(end-ResH),: ), SLExcludeHoriWei, [] ));
    NuRow = size(XI,2);
    YI = repmat(InitialRow:(InitialRow+NuRow-1), [SLExcludeHoriWei 1]);
    InitialRow = InitialRow+NuRow;
    SH(sub2ind(size(SH),YI(:),XI(:))) = 1;
end
SH(:,end) = [];
mask = sum(SH,2) ==0;
SH(mask,:) = [];

% make sure diagnal terms are 1
DiInd = sub2ind([NuNei NuNei],(1:NuNei)',(1:NuNei)');
WI = WI+WI'; % make symmetric
WL = WL+WL'; % make symmetric
WIL = WIL+WIL'; % make symmetric
WI(DiInd) = 1;
WL(DiInd) = 1;
WIL(DiInd) = 1;

% maping Laser occlusion estimation to Bp matrix
[Bp, OccluMap, BoundaryLaserOccluHori, BoundaryLaserOccluVert] = LaserDetectOcc2Bp( LaserDepth, BpWidthV, BpWidthH, BoundaryPVert, BoundaryPHori, Hash2Ind);
%[BoundaryLaserOccluHori, BoundaryLaserOccluVert ] = BHash2BMap(Bp, BoundaryPHori, BoundaryPVert, ClosestNHashList); % output two binary mask

% Show Laser occlusion detection labeled Boundary
figure(100); subplot(2,2,2);
ImgMapOcclu = Img;
ImgMapOcclu(:,:,1) = 255*imresize(OccluMap, [ SegVertYSize SegHoriXSize]);
image(ImgMapOcclu);
PlotGridBoundary( BoundaryLaserOccluHori, BoundaryLaserOccluVert, VertexX, VertexY, [ SegVertYSize SegHoriXSize], 100,'g');
save([ScratchDataFolder '/Temp/OccluMRF' num2str(k) '.mat']);

% Run MRF
[U S V] = svds(WI);
Wsqrt = (S.^(0.5))*V';
opt = sdpsettings('solver','sedumi');
B = sdpvar(NuNei,1);
t = sdpvar(1,1);
%F = set( -1 < B < 1) + set(cone(Wsqrt*B,t));
F = set( -1 < B < 1) + set(cone(Wsqrt*B,t))+set(SV*B <= (-sum(SV,2)+2) )+set(SH*B <= (-sum(SH,2)+2) );
beta = 1; % trade off weight
sol = solvesdp(F,t*t - beta*Bp'*B ,opt);
B = double(B);
toc
% beta = 1;
% cvx_begin
%     variable B(NuNei);
%     minimize(B'*WI*B - beta*Bp'*B)
%     abs(B) <= 1;
%     SV*B <= (-sum(SV,2)+2);
%     SH*B <= (-sum(SH,2)+2);
% cvx_end
[BoundaryMRFEstHori, BoundaryMRFEstVert ] = BHash2BMap(B, BoundaryPHori, BoundaryPVert, ClosestNHashList); % output two binary mask

% Show Laser occlusion detection labeled Boundary
figure(100); subplot(2,2,3);
image(ImgMapOcclu);
PlotGridBoundary( BoundaryMRFEstHori, BoundaryMRFEstVert, VertexX, VertexY,[ SegVertYSize SegHoriXSize], 100,'y');
saveas(100,[ScratchDataFolder '/data/occlu/MRFoccl' num2str(k) '.jpg']);
close all;
