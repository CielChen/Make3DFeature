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
function [vrml_filename] = vrml_test_faceset_triangle(filename,Position3D,depth,ray,DepthDirectory,HBrokeBook,VBrokeBook,Grid,maskSky,Np,Sep,a,b,Ox,Oy,Thresh)
%function [] = vrml_test_faceset_triangle(filename,PlaneParameterTure,LowResImgIndexSuperpixelSep,DepthDirectory,a,b,Ox,Oy)
% this function play with the VRML
% using only FaceSet
% % global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
% %     ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
% %     Horizon_default batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

if nargin < 12
    a = 0.70783777 %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
    b = 0.946584169%0.085; % vertical physical size of image plane normalized to focal length (in meter)
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 13
    b = a;
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 14
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 15
    Oy = Ox;
end

% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder;

vrml_filename = ['tri_' filename '_' 'flipTri4.wrl'];
%LowResImgIndexSuperpixelSepTemp = LowResImgIndexSuperpixelSep;
%[VertYSize HoriXSize] = size(LowResImgIndexSuperpixelSepTemp);
[dum VertYSize HoriXSize] = size(Position3D);
%nu_patch = VertYSize* HoriXSize;

Position3DCoord = reshape(Position3D,3,VertYSize*HoriXSize);    
Position3DCoord(3,:) = -Position3DCoord(3,:); % important to make z direction negative

table = reshape(0:(VertYSize*HoriXSize-1),VertYSize,[]);
Position3DCoordOrdeerIndex = [];
GridPosition3DCoordOrdeerIndex = [];

countPoints=length(Position3DCoord)+1;
unassignedTargets = 0;
for i = 1:VertYSize-1
    for j = 1:HoriXSize-1
        % check which diagnoal to render
        target = [HBrokeBook(i,j) VBrokeBook(i,j+1) HBrokeBook(i+1,j) VBrokeBook(i,j)];
        clear Index;
        clear GridIndex;
        clear indexArray;
        
        if (all(target == [0 0 0 0]) || all(target == [0 0 1 1]) || all(target == [1 1 0 0]))
            Index = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i+1,j+1) table(i,j+1)];
            GridIndex = [table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i,j) table(i+1,j+1) table(i,j+1) table(i,j)];
            indexArray = [i i+1 i+1 i i+1 i; ...
                          j j   j+1 j j+1 j+1];
        elseif (all(target == [1 0 0 1]) || all(target == [0 1 1 0]))
            Index = [ table(i,j) table(i+1,j) table(i,j+1) table(i+1,j) table(i+1,j+1) table(i,j+1)];
            GridIndex = [ table(i,j) table(i+1,j) table(i,j+1) table(i,j) table(i+1,j) table(i+1,j+1) table(i,j+1) table(i+1,j)];
            indexArray = [i i+1 i   i+1 i+1 i; ...
                          j j   j+1 j   j+1 j+1];
        else
            unassignedTargets=unassignedTargets+1;
            Index = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i+1,j+1) table(i,j+1)];
            GridIndex = [table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i,j) table(i+1,j+1) table(i,j+1) table(i,j)];
            indexArray = [i i+1 i+1 i i+1 i; ...
                          j j   j+1 j j+1 j+1];
        end

        % Sky Exclude 
        if maskSky(i,j)==1
        %% check if the neighbors are sky as well, for Np length
            if (j+(Np-1))>HoriXSize
                jL=HoriXSize;
            else
                jL=j+(Np-1);
            end
            if (i+(Np-1))>VertYSize
                iL=VertYSize;
            else
                iL=i+(Np-1);
            end
            if ((sum(xor(maskSky(i,j:jL),ones(1,length(maskSky(i,j:jL)))))>0)||...
                    sum(xor(maskSky(i:iL,j),ones(length(maskSky(i:iL,j)),1)))>0)
                %% also update the depth values for all the 6 points by
                %% changing them to a value of any non-sky point in the
                %% vicinity of Np down and to the right

                %% so first we find the point non-sky point which made us
                %% include this sky point
                iRep=i;
                indexVec=j:jL;
                nonSkyPoints=find(maskSky(i,indexVec)==0);
                if length(nonSkyPoints)>0
                    jRep=indexVec(nonSkyPoints(1));
                else
                    jRep=j;
                    indexVec=i:iL;
                    nonSkyPoints=find(maskSky(indexVec,j)==0);
                    iRep=indexVec(nonSkyPoints(1));
                end
                iChoose=iRep;
                jChoose=jRep;
                firstTriPoints=[];
                num=1;
                [firstTriPoints,Position3DCoord, depth]=removeSkyPointsDepth(maskSky,indexArray,num,firstTriPoints,ray,depth,VertYSize,HoriXSize,Position3DCoord,iChoose,jChoose);                
                max_or_second_max=1;
                [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(firstTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                    VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
                max_or_second_max=2;
                [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(firstTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                    VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
%                 
                secondTriPoints=[];
                num=2;
                [secondTriPoints,Position3DCoord, depth]=removeSkyPointsDepth(maskSky,indexArray,num,secondTriPoints,ray,depth,VertYSize,HoriXSize,Position3DCoord,iChoose,jChoose);                
                max_or_second_max=1;
                [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(secondTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                    VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
                max_or_second_max=2;
                [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(secondTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                    VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);
% 		
		        Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index];
		        GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex];
%                 if ((max(Index(1:3))<16775)&&(max(GridIndex(1:4))<16775))
%                     Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index(1:3)];
%                     GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex(1:4)];
%                 end
%                 if ((max(Index(4:6))<16775)&&(max(GridIndex(5:8))<16775))
%                     Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index(4:6)];
%                     GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex(5:8)];
%                 end
            end
        else
            iChoose=i;
            jChoose=j;
            firstTriPoints=[];
            num=1;
            [firstTriPoints,Position3DCoord, depth]=removeSkyPointsDepth(maskSky,indexArray,num,firstTriPoints,ray,depth,VertYSize,HoriXSize,Position3DCoord,iChoose,jChoose);                
            max_or_second_max=1;
            [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(firstTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
            max_or_second_max=2;
            [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(firstTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
%             
            secondTriPoints=[];
            num=2;
           [secondTriPoints,Position3DCoord, depth]=removeSkyPointsDepth(maskSky,indexArray,num,secondTriPoints,ray,depth,VertYSize,HoriXSize,Position3DCoord,iChoose,jChoose);                
            max_or_second_max=1;
            [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(secondTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);                
            max_or_second_max=2;
            [countPoints, Position3DCoord, Index, GridIndex, depth]=removeOffsetPointsReplaceDepth(secondTriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                VertYSize, HoriXSize, Position3DCoord, Index, GridIndex, countPoints);
% 
            Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index];
            GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex];

%             if ((max(Index(1:3))<16775)&&(max(GridIndex(1:4))<16775))
%                 Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index(1:3)];
%                 GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex(1:4)];
%             end
%             if ((max(Index(4:6))<16775)&&(max(GridIndex(5:8))<16775))
%                 Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index(4:6)];
%                 GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex(5:8)];
%             end
        end
%         Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index];
%         GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex];
    end
end

% calculate PositionTexCoord
PositionTex = permute(ray(:,:,1:2)./repmat(cat(3,a,b),[VertYSize HoriXSize 1])+0.5,[3 1 2]);
PositionTexCoord = PositionTex;   

Xmin=min(Position3DCoord(1,:))-Sep;
Xmax=max(Position3DCoord(1,:))+Sep;
Ymin=min(Position3DCoord(2,:))-Sep;
Ymax=max(Position3DCoord(2,:))+Sep;
Zmin=min(Position3DCoord(3,:))-Sep;
Zmax=min(Position3DCoord(3,:))+Sep;
%% the eight corners of the cube would then be
cc=[];
cc = [cc [Xmin Ymin Zmin]']; %% 1
cc = [cc [Xmax Ymin Zmin]']; %% 2
cc = [cc [Xmin Ymin Zmax]']; %% 3
cc = [cc [Xmax Ymin Zmax]']; %% 4

cc = [cc [Xmin Ymax Zmin]']; %% 5
cc = [cc [Xmax Ymax Zmin]']; %% 6
cc = [cc [Xmin Ymax Zmax]']; %% 7
cc = [cc [Xmax Ymax Zmax]']; %% 8

% wallIndices = [front_wall right_wall back_wall left_wall];
wallIndices = [2 4 8 6 4 8 7 3 3 7 5 1]-1;

% inital header
disp('writing vrml..');
fp = fopen([ScratchDataFolder '/vrml/' vrml_filename],'w');

fprintf(fp, '#VRML V2.0 utf8\n');

% add navigate_info
fprintf(fp, 'NavigationInfo {\n');
fprintf(fp, '  headlight TRUE\n');
fprintf(fp, '  type ["FLY", "ANY"]}\n\n');

% add viewpoint
fprintf(fp, 'Viewpoint {\n');
fprintf(fp, '    position        0 0.0 0.0\n');
fprintf(fp, '    orientation     0 0 0 0\n');
fprintf(fp, '    fieldOfView     0.7\n');
fprintf(fp, '    description "Original"}\n');

% add Shape for texture faceset
fprintf(fp, 'Shape{\n');
fprintf(fp, '  appearance Appearance {\n');
fprintf(fp, ['   texture ImageTexture { url "./image/' filename '.jpg' '" }\n']);
fprintf(fp, '  }\n');
fprintf(fp, '  geometry IndexedFaceSet {\n');
fprintf(fp, '    coord Coordinate {\n');

% insert coordinate in 3d
% =======================
fprintf(fp, '      point [ \n');
fprintf(fp, '        %.2f %.2f %.2f,\n',Position3DCoord);
fprintf(fp, '      ]\n');
fprintf(fp, '    }\n');

% insert coordinate index in 3d
fprintf(fp, '    coordIndex [\n');
fprintf(fp, '              %g %g %g -1,\n',Position3DCoordOrdeerIndex);
fprintf(fp, '    ]\n');

% insert texture coordinate
fprintf(fp, '    texCoord TextureCoordinate {\n');
fprintf(fp, '      point [\n');
fprintf(fp, '              %.4g %.4g,\n',PositionTexCoord);
fprintf(fp, '        ]\n');
fprintf(fp, '    }\n');
fprintf(fp, '  }\n');
fprintf(fp, '}\n');
% %% Now we make the wall
% 
% % add Shape for texture faceset
% fprintf(fp, 'Shape{\n');
% fprintf(fp, '  appearance Appearance {\n');
% %% one can change the sky color here, at the moment its a shade of blue
% %% given as [0.31 0.54 0.76]
% fprintf(fp, '  material Material { emissiveColor 0.31 0.54 0.76 }\n'); 
% fprintf(fp, '  }\n');
% fprintf(fp, '  geometry IndexedFaceSet {\n');
% fprintf(fp, '    coord Coordinate {\n');
% 
% % insert coordinate in 3d
% % =======================
% fprintf(fp, '      point [ \n');
% fprintf(fp, '        %.2f %.2f %.2f,\n',cc);
% fprintf(fp, '      ]\n');
% fprintf(fp, '    }\n');
% 
% % insert coordinate index in 3d
% fprintf(fp, '    coordIndex [\n');
% %Position3DCoordOrdeerIndex = 0:(size(Position3DCoord,2)-1);
% fprintf(fp, '              %g %g %g %g -1,\n',wallIndices);
% fprintf(fp, '    ]\n');
% 
% % ==================================
% fprintf(fp, '  }\n');
% fprintf(fp, '}\n');

if Grid == 1
% ========================================
fprintf(fp, 'Shape{\n');
fprintf(fp, '  appearance Appearance { material Material {emissiveColor 1 0 0  }}\n');
fprintf(fp, '	 geometry IndexedLineSet {\n');
fprintf(fp, '    coord Coordinate {\n');
fprintf(fp, '      point [ \n');
fprintf(fp, '        %.2f %.2f %.2f,\n',Position3DCoord);
fprintf(fp, '      ]\n');
fprintf(fp, '    }\n');
fprintf(fp, '    coordIndex [\n');
fprintf(fp, '              %g %g %g %g -1,\n',GridPosition3DCoordOrdeerIndex);
fprintf(fp, '    ]\n');
%fprintf(fp, '	 color Color { color [1 0 0, 0 1 0, 0 0 1]}\n');
%fprintf(fp, '	 colorIndex [0 0 0 0 0 ]\n');
fprintf(fp, '	 colorPerVertex FALSE\n');
fprintf(fp, '	 }\n');
%==================================
fprintf(fp, '  }\n');
fprintf(fp, '}\n');
end
% close the file
fclose(fp);

