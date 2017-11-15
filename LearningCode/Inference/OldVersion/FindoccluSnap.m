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
function []=FindOccluSnap(k, Ray, RayPorjectImgMapX, RayPorjectImgMapY) % for each image

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

% inital parameter setting
ThreSize = 10;
StickHori = 0;
StickVert = 0;
ClosestDist = 1;
FarestDist = 80;
ThreUpDown = 0.5;
ThreOcclu = 1;
SE = strel('disk',3);
SE5 = strel('disk',5);
ThreVert = 0.5;
ThreHori = 0.5;
ThreFar = 15;
ThrePcik = 0.5;
ThreOut = 0.3;

% get all the data we need laserDepth Sup MedSup
depthfile = strrep(filename{k},'img','depth_sph_corr'); % the depth filename
load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
LaserDepth = Position3DGrid(:,:,4); 
clear Position3DGrid;
 % load the CleanedSup
load([ScratchDataFolder '/data/CleanSup/CleanSup' num2str(k) '.mat']);         
[SegVertYSize, SegHoriXSize] = size(MedSup);
MedSup = double(MedSup);
MaxMedSup = max(MedSup(:));
Sup = double(Sup);
 % generate the Posi3D
Posi3D = im_cr2w_cr(LaserDepth,permute(Ray,[2 3 1]));
 % generate the RayPorjectImgMapY and RayPorjectImgMapX to resolution as MedSup
NewRayPorjectImgMap = Matrix2ImgCo(HoriXNuDepth, VertYNuDepth, [RayPorjectImgMapX(:) RayPorjectImgMapY(:)]);
NewRayPorjectImgMap = ImgCo2Matrix(SegHoriXSize, SegVertYSize, NewRayPorjectImgMap);
NewRayPorjectImgMap = reshape(NewRayPorjectImgMap',[2 VertYNuDepth HoriXNuDepth]);
 % read in img
Img = imresize(imread([GeneralDataFolder '/' ImgFolder '/' filename{k} ],'jpg'), [SegVertYSize SegHoriXSize]);

% generate Straight lines
[seglist]=edgeSegDetection(Img,k,0);
Pointer = seglist(:,1) > seglist(:,3);
temp = seglist(Pointer,1:2);
seglist(Pointer,1:2) = seglist(Pointer,3:4);
seglist(Pointer,3:4) = temp;
edgeIm = zeros(SegVertYSize, SegHoriXSize);
NuSeg =size(seglist,1);
for l = 1:NuSeg
    l
    x = (floor(seglist(l,1)):ceil(seglist(l,3)))';
    if seglist(l,1) ~= seglist(l,3)
       y = round(LineProj(seglist(l,:), x , []));
    else
        if seglist(l,2) < seglist(l,4)
           y = (floor(seglist(l,2)):ceil(seglist(l,4)))';
        else
           y = (floor(seglist(l,4)):ceil(seglist(l,2)))' ;
        end
        x = round(LineProj(seglist(l,:), [] , y));
    end
    index = sub2ind([SegVertYSize SegHoriXSize],y,x);
    edgeIm(index) = l;
    
    if seglist(l,2) < seglist(l,4)
           y = (floor(seglist(l,2)):ceil(seglist(l,4)))';
    else
           y = (floor(seglist(l,4)):ceil(seglist(l,2)))' ;
    end
    if seglist(l,2) ~= seglist(l,4)
      x = round(LineProj(seglist(l,:), [] , y));
    else
      x = (floor(seglist(l,1)):ceil(seglist(l,3)))';
      y = round(LineProj(seglist(l,:), x , [])); 
    end
    index = sub2ind([SegVertYSize SegHoriXSize],y,x);
    edgeIm(index) = l;
end
figure(400); imagesc(edgeIm);

% Find Spatial Jump Point
DiffDepthVert = abs(conv2(LaserDepth,[1; -1],'valid'));
DiffDepthHori = abs(conv2(LaserDepth,[1 -1],'valid'));
FraDiffDepthVert = DiffDepthVert./ sqrt(LaserDepth(1:(end-1),:) .* LaserDepth(2:end,:) );
%FraDiffDepthVert = DiffDepthVert./ min(LaserDepth(1:(end-1),:) , LaserDepth(2:end,:) );
OccFraDiffDepthVert = FraDiffDepthVert > ThreVert;
OccFraDiffDepthVert(LaserDepth(1:(end-1),:) > ThreFar & LaserDepth(2:end,:) > ThreFar) = 0;
FraDiffDepthHori = DiffDepthHori./ sqrt( LaserDepth(:,1:(end-1)) .* LaserDepth(:,2:end) );
%FraDiffDepthHori = DiffDepthHori./ min( LaserDepth(:,1:(end-1)) , LaserDepth(:,2:end) );

OccFraDiffDepthHori = FraDiffDepthHori > ThreHori;
OccFraDiffDepthHori(LaserDepth(:,1:(end-1)) > ThreFar & LaserDepth(:,2:end) > ThreFar) = 0;
OccFarDiffDepthMask = zeros(VertYNuDepth, HoriXNuDepth);
OccFarDiffDepthMask(:,1:(end-1)) = OccFraDiffDepthHori;
OccFarDiffDepthMask(:,2:(end)) = OccFarDiffDepthMask(:,2:end) | OccFraDiffDepthHori;
OccFarDiffDepthMask(1:(end-1),:) = OccFarDiffDepthMask(1:(end-1),:) | OccFraDiffDepthVert;
OccFarDiffDepthMask(2:(end),:) = OccFarDiffDepthMask(2:end,:) | OccFraDiffDepthVert;
figure(1000);
subplot(3,2,1);
TempDepth = LaserDepth;
TempDepth(logical(OccFarDiffDepthMask)) =-10;
imagesc(TempDepth);
title('LaserDepthMap');
figure(1000);
subplot(3,2,2);
ii = Img;
ii(:,:,1) = 255*logical(edgeIm);
image(ii);
title('edgeDetected');

% show all the possible candidate pairs
MedBoundaryPHori = conv2(MedSup,[1 -1],'same') ~=0;
MedBoundaryPHori(:,end) = 0;
MedBoundaryPVert = conv2(MedSup,[1; -1],'same') ~=0;
MedBoundaryPVert(end,:) = 0;
MedBoundaryP = MedBoundaryPVert |MedBoundaryPHori;
ClosestNList = [ MedSup(find(MedBoundaryPHori==1)) MedSup(find(MedBoundaryPHori==1)+SegVertYSize);...
                 MedSup(find(MedBoundaryPVert==1)) MedSup(find(MedBoundaryPVert==1)+1)];
ClosestNList = sort(ClosestNList,2);
ClosestNList = unique(ClosestNList,'rows');
ClosestNList(ClosestNList(:,1) == 0,:) = [];
NuPair = size(ClosestNList,1)
%generate the boundary index mask
BoundaryIndexMask = zeros(SegVertYSize, SegHoriXSize);
HashList = ClosestNList(:,1)*MaxMedSup+ClosestNList(:,2);
lH = find(MedBoundaryPHori);
j = lH + SegVertYSize;
HashBoundaryHori = sort([MedSup(lH) MedSup(j)],2);
HashBoundaryHori = HashBoundaryHori(:,1)*MaxMedSup + HashBoundaryHori(:,2);
lV = find(MedBoundaryPVert);
j = lV + 1;
HashBoundaryVert = sort([MedSup(lV) MedSup(j)],2);
HashBoundaryVert = HashBoundaryVert(:,1)*MaxMedSup + HashBoundaryVert(:,2);
for m = 1:size(HashList,1)
    Pointer = HashBoundaryHori == HashList(m);
    BoundaryIndexMask(lH(Pointer)) = m;
    Pointer = HashBoundaryVert == HashList(m);
    BoundaryIndexMask(lV(Pointer)) = m;
end

figure(200); imagesc(BoundaryIndexMask);

if false
figure(300); ii = Img;
ii(:,:,1) = 255*MedBoundaryP;
image(ii);
hold on;
scatter(NewRayPorjectImgMap(1,logical(OccFarDiffDepthMask)),NewRayPorjectImgMap(2,logical(OccFarDiffDepthMask)),100,LaserDepth(logical(OccFarDiffDepthMask)));
hold off;
end

OccFarDiffDepthMask = imresize(OccFarDiffDepthMask, size(MedBoundaryP) );

Z_Mask = (MedBoundaryP .* imdilate(OccFarDiffDepthMask, SE5) ) | ( logical(edgeIm) .* imdilate(OccFarDiffDepthMask, SE5) );
temp_mask = (MedBoundaryP .* imdilate(OccFarDiffDepthMask, SE5) );
Mask_2nd = ( logical(edgeIm) .* imdilate(OccFarDiffDepthMask, SE5) );

%comlete the boundary
ComPMask = Z_Mask;
for l = 1:NuPair
    mask = BoundaryIndexMask == l;
    AllSize = sum(mask(:));
    AndMask = Z_Mask & mask;
    AndSize = sum(AndMask(:));
    PerCent = AndSize/AllSize;
    if PerCent > ThrePcik
       ComPMask(mask) = 1;
    elseif PerCent < ThreOut
       ComPMask(mask) = 0;
    end
end

% link CompMAsk
[seglistNew ]=edgeSegDetection_new(ComPMask,k,1, 5, 3, 0.01, 3, 20);

figure(1000); ii = Img;
subplot(3,2,3);
ii(:,:,1) = 255*MedBoundaryP;
ii(:,:,3) = 255*OccFarDiffDepthMask;
image(ii);
title('Dilate_Occlu');
hold on;

figure(700); ii = Img;
ii(:,:,1) = 255*temp_mask;
image(ii);
hold on;
figure(800); ii = Img;
ii(:,:,1) = 255*Mask_2nd;
image(ii);
hold on;

figure(600); ii = Img;
ii(:,:,1) = 255*Z_Mask;
image(ii);
hold on;

figure(1000); 
subplot(3,2,4);
ii = Img;
ii(:,:,1) = 255*ComPMask;
image(ii);
title('Complete_Occlu');
hold on;

figure(1000); 
subplot(3,2,5);
image(Img);
drawseg(seglistNew,1000);
title('LinkComplete');

saveas(1000,[ScratchDataFolder '/data/occlu/' filename{k} 'occlu_comp50.fig'],'fig');
%scatter(NewRayPorjectImgMap(1,logical(OccFarDiffDepthMask)),NewRayPorjectImgMap(2,logical(OccFarDiffDepthMask)),100,LaserDepth(logical(OccFarDiffDepthMask)));
hold off;

return;
