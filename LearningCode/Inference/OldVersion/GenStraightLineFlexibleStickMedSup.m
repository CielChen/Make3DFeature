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
function [Sup, MedSup, RayProjImgCo, MovedPatchBook, HBrokeBook, VBrokeBook, StraightLineTable, OccluList] = ...
          GenStraightLine(seglist,MedSup, Sup, RayProjImgCo, Depth, MovedPatchBook, HBrokeBook, VBrokeBook, Ox, Oy, a, b)

% This function 1)clean the Sup at the boundary of lines
% 2) Stick the Ray closest to the boundary
% 3) find out the StraightLineTable to be used as constrain in MRF
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;


% initalize parameters
NuSeg = size(seglist,1);
Htol = 	15*(pi/180);
Vtol =  15*(pi/180);
Vert3DTol = 80*(pi/180);
Hori3DTol = 80*(pi/180);
SptialDiff = 5;
slab = 0; % extend the Straight line prior to wider in a slab
StraightLineTable = [];
OccluList = [];
hardMerge = false;
debug = false;
Regroup = true;
YRayProjPosi = permute(RayProjImgCo(2,:,:),[2 3 1]); %[VertYNuDepth HoriXNuDepth]
XRayProjPosi = permute(RayProjImgCo(1,:,:),[2 3 1]); %[VertYNuDepth HoriXNuDepth]

% sort the seglist by the length of seg in decending order
List = sortrows([norms((seglist(:,1:2)-seglist(:,3:4))')' (1:NuSeg)'],1);
List = flipud(List);
seglist = seglist( List(:,end),:);
% arrange the seglist to have start point a smaller X
Pointer = seglist(:,1) > seglist(:,3);
temp = seglist(Pointer,1:2);
seglist(Pointer,1:2) = seglist(Pointer,3:4);
seglist(Pointer,3:4) = temp;
seglistMed = seglist;
seglist(:,1:2) = Matrix2ImgCo(SegHoriXSize, SegVertYSize, seglist(:,1:2));
seglist(:,3:4) = Matrix2ImgCo(SegHoriXSize, SegVertYSize, seglist(:,3:4));
seglist(:,1:2) = ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, seglist(:,1:2)); % now in the resolution of HoriXNuDepth, VertYNuDepth
seglist(:,3:4) = ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, seglist(:,3:4));

% Find Projection point from x grid and y grid (Ray Move with Y grid only)
%DownRightFlag = 1;
OrientFlag = '';
for i = 1:NuSeg
    % find the YProjXGrid
    if seglist(i,1) ~=seglist(i,3)
       XGrid = (floor(seglist(i,1)):ceil(seglist(i,3)))';% interger Extend to make sure no rendering across line
       XGridMed = Matrix2ImgCo(HoriXNuDepth, VertYNuDepth, [XGrid ones(size(XGrid,1),1)]);
       XGridMed = ImgCo2Matrix( SegHoriXSize, SegVertYSize, XGridMed);
       XGridMed(:,2) = [];
    else
       XGrid = [];
       XGridMed = [];
    end
    if ~isempty(XGrid)
       YProjXGridMed = LineProj(seglistMed(i,:),XGridMed,[]);% real number
    else
       YProjXGridMed = sparse(0,1);
    end
    % check if YProjXGrid > VertYNuDepth or YProjXGrid < 0

    if ~isempty(XGridMed)
       OutOfRange = XGridMed >= SegHoriXSize | XGridMed <= 1; % use it to sample the new Sup, os must be within range
       OutOfRange = OutOfRange | ( YProjXGridMed >= SegVertYSize | YProjXGridMed <= 1); % same reason
       YProjXGridMed(OutOfRange) = [];
       XGridMed(OutOfRange) = [];
       XGrid(OutOfRange) = [];
    end

    % find the YXProjYGrid
    if seglist(i,2) > seglist(i,4)
       YGrid = (floor(seglist(i,4)):ceil(seglist(i,2)))';% interger Extend to make sure no rendering across line
    elseif seglist(i,2) < seglist(i,4)
       YGrid = (floor(seglist(i,2)):ceil(seglist(i,4)))';% interger Extend to make sure no rendering across line
    else 
       YGrid = [];
       YGridMed = [];
    end
    if ~isempty(YGrid)
       YGridMed = Matrix2ImgCo(HoriXNuDepth, VertYNuDepth, [ones(size(YGrid,1),1) YGrid]);
       YGridMed = ImgCo2Matrix( SegHoriXSize, SegVertYSize, YGridMed);
       YGridMed(:,1) = [];
    end
    if ~isempty(YGridMed)
       XProjYGridMed = LineProj(seglistMed(i,:),[], YGridMed);% real number
    else
       XProjYGridMed = sparse(0,1);
    end
    % check if XProjYGrid > HoriXNuDepth or XProjYGrid < 0
    if ~isempty(YGridMed)
       OutOfRange = YGridMed >= SegVertYSize | YGridMed < 1; % use it to sample the new Sup, os must be within range
       OutOfRange = OutOfRange | (XProjYGridMed > SegHoriXSize | XProjYGridMed < 1); % same reason
       XProjYGridMed(OutOfRange) = [];
       YGridMed(OutOfRange) = [];
       YGrid(OutOfRange) = [];
    end

    % find Point in two side
    if ~isempty(XGrid)
        mask = YRayProjPosi(:,XGrid) < repmat(YProjXGridMed',[size(YRayProjPosi,1) 1]);
        [YProjXGridMed_U YSub_U] = max(YRayProjPosi(:,XGrid).*mask,[],1);
        YProjXGridMed_U =YProjXGridMed_U';
        YSub_U = YSub_U';
        mask = YRayProjPosi(:,XGrid) >= repmat(YProjXGridMed',[size(YRayProjPosi,1) 1]);
        mask = double(mask);
        mask(~mask) = Inf;
        [YProjXGridMed_D YSub_D] = min(YRayProjPosi(:,XGrid).*mask,[],1);
        YProjXGridMed_D = YProjXGridMed_D';
        YSub_D = YSub_D';
        % make sure YSub_D > YSub_U
        mask = YSub_D <= YSub_U;
        YProjXGridMed_D(mask) = [];
        YProjXGridMed_U(mask) = [];
        YSub_U(mask) = [];
        YSub_D(mask) = [];
        XGrid(mask) = [];
        XGridMed(mask) = [];
        YProjXGridMed(mask) = [];
    end
    if ~isempty(YGrid)
        mask = XRayProjPosi(YGrid,:) < repmat(XProjYGridMed,[1 size(XRayProjPosi,2) ]);
        [XProjYGridMed_L XSub_L] = max(XRayProjPosi(YGrid,:).*mask,[],2);
        mask = XRayProjPosi(YGrid,:) >= repmat(XProjYGridMed,[1 size(XRayProjPosi,2) ]);
        mask = double(mask);
        mask(~mask) = Inf;
        [XProjYGridMed_R XSub_R] = min(XRayProjPosi(YGrid,:).*mask,[],2);
        % make sure XSub_R > XSub_L
        mask = XSub_R <= XSub_L;
        XProjYGridMed_R(mask) = [];
        XProjYGridMed_L(mask) = [];
        XSub_L(mask) = [];
        XSub_R(mask) = [];
        YGrid(mask) = [];
        YGridMed(mask) = [];
        XProjYGridMed(mask) = [];
    end

    % Stick Ray to segline boundary moving in X dorection only
    % check MovedPatchBook before move
    MovedPatchBookUpdate = MovedPatchBook;
    if ~isempty(XGrid) % move Y derection and check if broke found
       index = sub2ind([ VertYNuDepth HoriXNuDepth ],YSub_U, XGrid);
       [cU,kU] = setdiff(index,MovedPatchBook);
       if ~isempty(cU) % move only is not moved before
%           if any(YProjXGridMed(kU) < YRayProjPosi( max(cU-1,1))) || ...
%              any(YProjXGridMed(kU) > YRayProjPosi( min(cU+1,VertYNuDepth*HoriXNuDepth)))
%              disp('error CU');
%              return;
%           end
          MovedPatchBookUpdate = [MovedPatchBookUpdate; cU];
          YRayProjPosi( cU) = YProjXGridMed(kU);
          classifiedU = LineProj(seglistMed(i,:),XGridMed(1), YProjXGridMed_U(1));
          if LineProj(seglistMed(i,:),floor(XRayProjPosi(cU)), floor(YRayProjPosi(cU))) == classifiedU
             Sup(cU) = MedSup(sub2ind([SegVertYSize SegHoriXSize],floor(YRayProjPosi(cU)),floor(XRayProjPosi(cU)))); 
          else
             Sup(cU) = MedSup(sub2ind([SegVertYSize SegHoriXSize],floor(YRayProjPosi(cU)),ceil(XRayProjPosi(cU)))); 
          end
          if classifiedU == 1
             StraightLine{1} = setdiff(Sup(cU)',0);
          else
             StraightLine{2} = setdiff(Sup(cU)',0);
          end
       end
       index = sub2ind([ VertYNuDepth HoriXNuDepth ],YSub_D, XGrid);
       [cD,kD] = setdiff(index,MovedPatchBook);
       if ~isempty(cD) % move only is not moved before
%           if any(YProjXGridMed(kD) < YRayProjPosi( max(cD-1,1))) || ...
%              any(YProjXGridMed(kD) > YRayProjPosi( min(cD+1,VertYNuDepth*HoriXNuDepth)))
%              disp('error CD');
%              return;
%           end
          MovedPatchBookUpdate = [MovedPatchBookUpdate; cD];
          YRayProjPosi( cD) = YProjXGridMed(kD);
          classifiedD = LineProj(seglistMed(i,:),XGridMed(1), YProjXGridMed_D(1));
          if LineProj(seglistMed(i,:),floor(XRayProjPosi(cD)), ceil(YRayProjPosi(cD))) == classifiedD 
             Sup(cD) = MedSup(sub2ind([SegVertYSize SegHoriXSize],ceil(YRayProjPosi(cD)),floor(XRayProjPosi(cD)))); 
          else
             Sup(cD) = MedSup(sub2ind([SegVertYSize SegHoriXSize],ceil(YRayProjPosi(cD)),ceil(XRayProjPosi(cD)))); 
          end
          if classifiedD == 1
             StraightLine{1} = setdiff(Sup(cD)',0);
          else
             StraightLine{2} = setdiff(Sup(cD)',0);
          end
       end
       % detect broken for rendering purpose
       VBrokeBook( index(intersect(kU,kD))) = 1;
      StraightLine{1} = unique(StraightLine{1});
      StraightLine{2} = unique(StraightLine{2});
      StraightLineTable = [StraightLineTable; (StraightLine)];
       
    end
    if ~isempty(YGrid)
       index = sub2ind([ VertYNuDepth HoriXNuDepth ],YGrid, XSub_L);
       [cL,kL] = setdiff(index,MovedPatchBook);
       [cLUpdate,kLUpdate] = setdiff(index,MovedPatchBookUpdate);
       if ~isempty(cLUpdate) % move only is not moved before
%           if any(XProjYGridMed(kLUpdate) < XRayProjPosi( max(cLUpdate-VertYNuDepth,1))) || ...
%              any(XProjYGridMed(kLUpdate) > XRayProjPosi( min(cLUpdate+VertYNuDepth,VertYNuDepth*HoriXNuDepth)))
%              disp('error CL');
%              return;
%           end
          MovedPatchBookUpdate = [MovedPatchBookUpdate; cLUpdate];
          XRayProjPosi( cLUpdate) = XProjYGridMed(kLUpdate);    
          classifiedL = LineProj(seglistMed(i,:),XProjYGridMed_L(1), YGridMed(1));
          if LineProj(seglistMed(i,:),floor(XRayProjPosi(cL)), floor(YRayProjPosi(cL))) == classifiedL 
             Sup(cL) = MedSup(sub2ind([SegVertYSize SegHoriXSize],floor(YRayProjPosi(cL)),floor(XRayProjPosi(cL)))); 
          else
             Sup(cL) = MedSup(sub2ind([SegVertYSize SegHoriXSize],ceil(YRayProjPosi(cL)),floor(XRayProjPosi(cL)))); 
          end
          if classifiedL == 1
             StraightLine{1} = Sup(cL)';
          else
             StraightLine{2} = Sup(cL)';
          end
       end
       index = sub2ind([ VertYNuDepth HoriXNuDepth ],YGrid, XSub_R);
       [cR,kR] = setdiff(index,MovedPatchBook);
       [cRUpdate,kRUpdate] = setdiff(index,MovedPatchBookUpdate);
       if ~isempty(cRUpdate) % move only is not moved before
%           if any(XProjYGridMed(kRUpdate) < XRayProjPosi( max(cRUpdate-VertYNuDepth,1))) || ...
%              any(XProjYGridMed(kRUpdate) > XRayProjPosi( min(cRUpdate+VertYNuDepth,VertYNuDepth*HoriXNuDepth)))
%              disp('error CR');
%              return;
%           end
          MovedPatchBookUpdate = [MovedPatchBookUpdate; cRUpdate];
          XRayProjPosi( cRUpdate) = XProjYGridMed(kRUpdate);
          classifiedR = LineProj(seglistMed(i,:),XProjYGridMed_R(1), YGridMed(1));
          if LineProj(seglistMed(i,:),ceil(XRayProjPosi(cR)), floor(YRayProjPosi(cR))) == classifiedR 
             Sup(cR) = MedSup(sub2ind([SegVertYSize SegHoriXSize],floor(YRayProjPosi(cR)),ceil(XRayProjPosi(cR)))); 
          else
             Sup(cR) = MedSup(sub2ind([SegVertYSize SegHoriXSize],ceil(YRayProjPosi(cR)),ceil(XRayProjPosi(cR)))); 
          end
          if classifiedR == 1
             StraightLine{1} = Sup(cR)';
          else
             StraightLine{2} = Sup(cR)';
          end
       end
       % detect broken for rendering purpose
       % Hori is much tricky, most make sure end point double stick and 
       % if the Vert Stick corresponding to the side hori stick
       HBrokeBook( index(intersect(kL,kR))) = 1;
    end
    MovedPatchBook = MovedPatchBookUpdate;
% end of sticking ray and broking rendering point=========================================================
% Start cleaning the MedSup 
    % define group in two side
    if seglistMed(i,1) ~= seglistMed(i,3)
       XGridMedDense = (ceil(seglistMed(i,1)):floor(seglistMed(i,3)))';
    else
       XGridMedDense = [];
    end
    if ~isempty(XGridMedDense)
       YProjXGridMedDense = LineProj(seglistMed(i,:),XGridMedDense,[]);% real number
    else
       YProjXGridMedDense = sparse(0,1);
    end
    % check if YProjXGrid > VertYNuDepth or YProjXGrid < 0

    if ~isempty(XGridMedDense)
       OutOfRange = XGridMedDense >= SegHoriXSize+0.5 | XGridMedDense <= 0.5;
       OutOfRange = OutOfRange | (YProjXGridMedDense >= SegVertYSize+0.5 | YProjXGridMedDense <= 0.5);
       YProjXGridMedDense(OutOfRange) = [];
       XGridMedDense(OutOfRange) = [];
    end

    % find the YXProjYGrid
    if seglistMed(i,2) > seglistMed(i,4)
       YGridMedDense = (ceil(seglistMed(i,4)):floor(seglistMed(i,2)))';% interger Extend to make sure no rendering across line
    elseif seglistMed(i,2) < seglistMed(i,4)
       YGridMedDense = (ceil(seglistMed(i,2)):floor(seglistMed(i,4)))';% interger Extend to make sure no rendering across line
    else
       YGridMedDense = [];
    end
    if ~isempty(YGridMedDense)
       XProjYGridMedDense = LineProj(seglistMed(i,:),[], YGridMedDense);% real number
    else
       XProjYGridMedDense = sparse(0,1);
    end
    % check if XProjYGrid > HoriXNuDepth or XProjYGrid < 0
    if ~isempty(YGridMedDense)
       OutOfRange = YGridMedDense >= SegVertYSize+0.5 | YGridMedDense < 0.5;
       OutOfRange = OutOfRange | (XProjYGridMedDense > SegHoriXSize+0.5 | XProjYGridMedDense < 0.5);
       XProjYGridMedDense(OutOfRange) = [];
       YGridMedDense(OutOfRange) = [];
    end

    group1 = [];
    group2 = [];
    group1Slab = [];
    group2Slab = [];
       if ~isempty(XGridMedDense)
          if all(LineProj(seglistMed(i,:),XGridMedDense, floor(YProjXGridMedDense)) == 1)
          group1 = [ group1; sub2ind([ SegVertYSize SegHoriXSize ], floor(YProjXGridMedDense), XGridMedDense)]; %side 1
          group1Slab = group1(:,ones(slab,1)) + ones(size(group1,1),1)*[-1:-1:-slab];
          group1Slab = group1Slab(:);
          group2 = [ group2; sub2ind([ SegVertYSize SegHoriXSize ], ceil(YProjXGridMedDense), XGridMedDense)]; % side 0
          group2Slab = group2(:,ones(slab,1)) + ones(size(group2,1),1)*[1:1:slab];
          group2Slab = group2Slab(:);
%          disp('Not suprising')
          elseif all(LineProj(seglistMed(i,:), XGridMedDense, ceil(YProjXGridMedDense)) == 1)
          group1 = [ group1; sub2ind([ SegVertYSize SegHoriXSize ], ceil(YProjXGridMedDense), XGridMedDense)]; %side 1
          group1Slab = group1(:,ones(slab,1)) + ones(size(group1,1),1)*[-1:-1:-slab];
          group1Slab = group1Slab(:);
          group2 = [ group2; sub2ind([ SegVertYSize SegHoriXSize ], floor(YProjXGridMedDense), XGridMedDense)]; % side 0
          group2Slab = group2(:,ones(slab,1)) + ones(size(group2,1),1)*[1:1:slab];
          group2Slab = group2Slab(:);
%          disp('suprising');
          else
            disp('error in combining group1 2');
%            return;
          end 
       end
       if ~isempty(YGridMedDense)
          if all(LineProj(seglistMed(i,:),ceil(XProjYGridMedDense), YGridMedDense) == 1)
             group1 = [ group1; sub2ind([ SegVertYSize SegHoriXSize ], YGridMedDense, ceil(XProjYGridMedDense))];
             temp = group1(:,ones(slab,1)) + ones(size(group1,1),1)*[1:1:slab]*SegVertYSize;
             group1Slab = [group1Slab; temp(:)];
             group2 = [ group2; sub2ind([ SegVertYSize SegHoriXSize ], YGridMedDense, floor(XProjYGridMedDense))];
             temp = group2(:,ones(slab,1)) + ones(size(group2,1),1)*[-1:-1:-slab]*SegVertYSize;
             group2Slab = [group2Slab; temp(:)];
          elseif all(LineProj(seglistMed(i,:),floor(XProjYGridMedDense), YGridMedDense) == 1)
             group1 = [ group1; sub2ind([ SegVertYSize SegHoriXSize ], YGridMedDense, floor(XProjYGridMedDense))];
             temp = group1(:,ones(slab,1)) + ones(size(group1,1),1)*[-1:-1:-slab]*SegVertYSize;
             group1Slab = [group1Slab; temp(:)];
             group2 = [ group2; sub2ind([ SegVertYSize SegHoriXSize ], YGridMedDense, ceil(XProjYGridMedDense))];
             temp = group2(:,ones(slab,1)) + ones(size(group2,1),1)*[1:1:slab]*SegVertYSize;
             group2Slab = [group2Slab; temp(:)];
          else
             disp('error in combining group1 2');
%             return;
          end
       end
    group1Slab(group1Slab <=0 | group1Slab > (SegVertYSize*SegHoriXSize)) = [];
    group2Slab(group2Slab <=0 | group2Slab > (SegVertYSize*SegHoriXSize)) = [];
    group1 = [group1 ; group1Slab];
    group2 = [group2 ; group2Slab];
%     if size(group1,1)<=3
%        if debug
%           disp('empty group1');
%        end
%        continue;
%     end
    % modify the Sup index according to the new boundary found
 if Regroup
    FuzzySupPoint = intersect( MedSup(group1), MedSup(group2));   
    if debug
       size(FuzzySupPoint) 
    end
    for j = FuzzySupPoint'
        if j == 439
           disp('Merge')
        end
        DominateFlag = 0;
        mask = MedSup == j;
        [My Mx] = find(mask);
        classified = LineProj(seglistMed(i,:),Mx, My);
        if sum(classified == 1) > sum(classified == 0)
           DominateFlag = 1;
        end
        if ~any(classified ~= DominateFlag)
           if debug
              j
              i
              sum(MedSup(group1) == j)
              sum(MedSup(group2) == j)
           end
           disp('error in classified')
           save([ScratchDataFolder '/data/FuzzyFlag.mat'],'-v6');
%           return;
        end
        targetX = [Mx(classified ~= DominateFlag)];
        targetY = [My(classified ~= DominateFlag)];
        MedSup(sub2ind([SegVertYSize SegHoriXSize ], targetY, targetX)) = ...
            regroup(MedSup, targetY, targetX, ~DominateFlag, seglistMed(i,:));
    end 
  end
% end of cleaning MedSup at the boundary ========================================================
% ===============================================================================================
      % detect occlusion
      [y1 x1] = ind2sub([SegVertYSize SegHoriXSize],group1);
      group1 = Matrix2ImgCo(SegVertYSize, SegHoriXSize, [x1 y1]);
      group1 = round(ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, group1));
      group1 = unique(group1,'rows');
      group1 = sub2ind([ VertYNuDepth HoriXNuDepth], max(min(group1(:,2),VertYNuDepth),1), max(min(group1(:,1),HoriXNuDepth),1));  
      [y2 x2] = ind2sub([SegVertYSize SegHoriXSize],group2);
      group2 = Matrix2ImgCo(SegVertYSize, SegHoriXSize, [x2 y2]);
      group2 = round(ImgCo2Matrix(HoriXNuDepth, VertYNuDepth, group2));
      group2 = unique(group2,'rows');
      group2 = sub2ind([ VertYNuDepth HoriXNuDepth], max(min(group2(:,2),VertYNuDepth),1), max(min(group2(:,1),HoriXNuDepth),1));  
      if abs(median(Depth(group1))- median(Depth(group2))) >= SptialDiff
         [X,Y] = meshgrid(Sup(group1),Sup(group2));
         OccluList = [OccluList; sort([X(:) Y(:)],2)]; 
      end
%   end
end
OccluList = unique(OccluList,'rows');
RayProjImgCo = permute(cat(3, XRayProjPosi, YRayProjPosi), [3 1 2]); 
save([ScratchDataFolder '/data/Sup.mat'],'Sup','-v6');
return;
