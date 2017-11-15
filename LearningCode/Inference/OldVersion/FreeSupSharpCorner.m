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
function [CoPEstDepth CoPM1 CoPM2 EstDepHoriStick HoriStickM_i HoriStickM_j EstDepVertStick VertStickM_i VertStickM_j WeiCoP NewSup NewNuSup NewNuSupSize NewSup2Para FixPara VertStickPointInd HoriStickPointInd] ...
            = FreeSupSharpCorner( Sup, Ray, OccluList, WeiV, ShiftStick, StickHori, StickVert, ClosestDist, depthMap, ...
              MultiScaleSupTable, verticalParaInd, graoundParaInd, MultiScaleFlag, Sup2Para);

% This function detect the para needs to be fixed 
% and free the corresponding para near the corner
% InPut : PlnaePara, Sup, HoriStickPointInd, VertStickPointInd,
% SepPointMeasureHori, SepPointMeasureVert
% OutPut:
% StickVertFreeM StickHoriFreeM Sup2Para FixPara

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;
% initial parameter
MaxSupNu = max(Sup(:));
OriMaxSupNu = MaxSupNu;
%threSize = prod(size(Sup))*3e-4;


% free every sup if not Vert or Ground==========================
NuSup =  setdiff( unique(Sup)',0);
NuSup = sort(NuSup);
NuSupFree = setdiff( NuSup, NuSup([ verticalParaInd'; graoundParaInd']) );
NewSup = Sup;
for i = NuSupFree
    i;
    maskfree = Sup == i;
%  end
    % set all free point a new sup index 
    NuNewFree = sum(maskfree(:));
    NewSup(maskfree) = ((MaxSupNu+1):(MaxSupNu+NuNewFree));
    Sup(maskfree) = 0;
    MaxSupNu = (MaxSupNu+NuNewFree);
end

% Generate StickVertFreeM StickHoriFreeM Sup2Para FixPara
NewNuSup = setdiff(unique(NewSup)',0);
NewNuSup = sort(NewNuSup);
NewNuSupSize = size(NewNuSup,2);
NewSup2Para = sparse(1,max(NewSup(:)));
NewSup2Para(NewNuSup) = 1:NewNuSupSize;
FixPara = NewNuSup(NewNuSup <= OriMaxSupNu);

BounaryPHori = conv2(Sup,[1 -1],'same') ~=0;
BounaryPHori(:,end) = 0;
BounaryPVert = conv2(Sup,[1; -1],'same') ~=0;
BounaryPVert(end,:) = 0;
ClosestNList = [ Sup(find(BounaryPHori==1)) Sup(find(BounaryPHori==1)+VertYNuDepth);...
                 Sup(find(BounaryPVert==1)) Sup(find(BounaryPVert==1)+1)];
ClosestNList = sort(ClosestNList,2);
ClosestNList = unique(ClosestNList,'rows');
ClosestNList(ClosestNList(:,1) == 0,:) = [];
% find the boundary point that might need to be stick ot each other in Sup at Vert and Hori direction==========================================
HoriStickM_i = sparse(0,3*NewNuSupSize);
HoriStickM_j = sparse(0,3*NewNuSupSize);
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
    Target(1) = NewSup2Para(Sup(i));
    Target(2) = NewSup2Para(Sup(j));
    rayBoundary(:,1) =  Ray(:,i);
    rayBoundary(:,2) =  Ray(:,i);
    if MultiScaleFlag
          vector = (MultiScaleSupTable(Sup2Para(Sup(i)),2:end) == MultiScaleSupTable(Sup2Para(Sup(j)),2:end)); % MultiScaleSupTable Worked in old index
          expV = exp(-10*(WeiV*vector' + ShiftStick) );
       betaTemp = StickHori*(0.5+1/(1+expV)); %*(DistStickLengthNormWei.^2)*beta(Target(I));
       % therr should always be sticking (know don't care about occlusion)
    else
       betaTemp = StickHori;
    end
    temp = sparse(3,NewNuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    HoriStickM_i = [HoriStickM_i; betaTemp*temp(:)'];
    temp = sparse(3,NewNuSupSize);
    temp(:,Target(2)) = rayBoundary(:,2);
    HoriStickM_j = [HoriStickM_j; betaTemp*temp(:)'];
    EstDepHoriStick = [EstDepHoriStick; sqrt(max(depthMap(i),ClosestDist)*max(depthMap(j),ClosestDist))];
    HoriStickPointInd = [HoriStickPointInd i ];
%  else
%    disp('Occlu');
%  end
end
VertStickM_i = sparse(0,3*NewNuSupSize);
VertStickM_j = sparse(0,3*NewNuSupSize);
VertStickPointInd = [];
EstDepVertStick = [];
for i = find(BounaryPVert==1)'
    j = i+1;
    if Sup(i) == 0 || Sup(j) == 0
       continue;
    end
%  if ~OccluList(sum( ClosestNList == repmat(sort([Sup(i) Sup(j)]), [NuNei  1]),2) == 2)
%  if ~any(sum( OccluList == repmat(sort([Sup(i) Sup(j)]), [size(OccluList,1)  1]),2) == 2)
    Target(1) = NewSup2Para(Sup(i));
    Target(2) = NewSup2Para(Sup(j));
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
    temp = sparse(3,NewNuSupSize);
    temp(:,Target(1)) = rayBoundary(:,1);
    VertStickM_i = [VertStickM_i; betaTemp*temp(:)'];
    temp = sparse(3,NewNuSupSize);
    temp(:,Target(2)) = rayBoundary(:,2);
    VertStickM_j = [VertStickM_j; betaTemp*temp(:)'];
    EstDepVertStick = [EstDepVertStick; sqrt(max(depthMap(i),ClosestDist)*max(depthMap(j),ClosestDist))];
    VertStickPointInd = [VertStickPointInd i ];
%  else
%    disp('Occlu');
%  end
end

BounaryPHori = conv2(NewSup,[1 -1],'same') ~=0;
BounaryPHori(:,end) = 0;
BounaryPVert = conv2(NewSup,[1; -1],'same') ~=0;
BounaryPVert(end,:) = 0;
ClosestNList = [ NewSup(find(BounaryPHori==1)) NewSup(find(BounaryPHori==1)+VertYNuDepth);...
                 NewSup(find(BounaryPVert==1)) NewSup(find(BounaryPVert==1)+1)];
ClosestNList = sort(ClosestNList,2);
ClosestNList = unique(ClosestNList,'rows');
ClosestNList(ClosestNList(:,1) == 0,:) = [];

% find Coplane need for New added Sup index only
NuNei = size(ClosestNList,1);
CoPM1 = sparse(0,3*NewNuSupSize);
CoPM2 = sparse(0,3*NewNuSupSize);
CoPEstDepth = sparse(0,0);
WeiCoP = [];
for i = 1: NuNei
%  if ~CornerList(i)
    if any( FixPara == ClosestNList(i,1)) &&  any( FixPara == ClosestNList(i,2)) 
       continue;
    end
    mask = NewSup == ClosestNList(i,1);
    SizeMaskAll = sum(mask(:));
    [y x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(y));
    y = find(mask(:,CenterX));
    if ~isempty(y)
       CenterY = round(median(y));
    end   
    temp1 = sparse(1, 3*NewNuSupSize);
    temp2 = sparse(1, 3*NewNuSupSize);
    temp1(:,(NewSup2Para(ClosestNList(i,1))*3-2): NewSup2Para(ClosestNList(i,1))*3) = Ray(:,CenterY,CenterX)';
    temp2(:,(NewSup2Para(ClosestNList(i,2))*3-2): NewSup2Para(ClosestNList(i,2))*3) = Ray(:,CenterY,CenterX)';
    wei = 1;%WeiV;

    CoPM1 = [CoPM1; temp1*wei];
    CoPM2 = [CoPM2; temp2*wei];
    tempWeiCoP = [SizeMaskAll];
    CoPEstDepth = [CoPEstDepth; max(median(depthMap(mask)),ClosestDist)];   
 
    mask = NewSup == ClosestNList(i,2);
    SizeMaskAll = sum(mask(:));
    [y x] = find(mask);
    CenterX = round(median(x));
    CenterY = round(median(y));
    y = find(mask(:,CenterX));
    if ~isempty(y)
       CenterY = round(median(y));
    end

    temp1 = sparse(1, 3*NewNuSupSize);
    temp2 = sparse(1, 3*NewNuSupSize);
    temp1(:,(NewSup2Para(ClosestNList(i,1))*3-2): NewSup2Para(ClosestNList(i,1))*3) = Ray(:,CenterY,CenterX)';
    temp2(:,(NewSup2Para(ClosestNList(i,2))*3-2): NewSup2Para(ClosestNList(i,2))*3) = Ray(:,CenterY,CenterX)';
    CoPM1 = [CoPM1; temp1*wei];
    CoPM2 = [CoPM2; temp2*wei];
    tempWeiCoP = [tempWeiCoP; SizeMaskAll];
    WeiCoP = [WeiCoP; tempWeiCoP];
    CoPEstDepth = [CoPEstDepth; max(median(depthMap(mask)),ClosestDist)];
%  end
end%=========================================================================================================
return;


