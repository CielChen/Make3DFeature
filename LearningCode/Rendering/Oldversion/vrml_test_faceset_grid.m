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
function [vrml_filename] = vrml_test_faceset_grid(filename,Position3D,depth,ray,DepthDirectory,HBrokeBook,VBrokeBook,Grid,maskSky,maskG, Np,Sep,a,b,Ox,Oy)
%function [] = vrml_test_faceset_triangle(filename,PlaneParameterTure,LowResImgIndexSuperpixelSep,DepthDirectory,a,b,Ox,Oy)
% this function play with the VRML
% using only FaceSet

displayFlag = true;

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize;

if nargin < 13
    a = 0.70783777 %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
    b = 0.946584169%0.085; % vertical physical size of image plane normalized to focal length (in meter)
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 14
    b = a;
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 15
    Ox = -0.010727086; % camera origin offset from the image center in horizontal direction
    Oy = -0.0111130176; % camera origin offset from the image center in vertical direction
elseif nargin < 16
    Oy = Ox;
end

if displayFlag
disp('In VRML generation Code');
end

% define global variable
global GeneralDataFolder ScratchDataFolder LocalFolder;

%maskSky = imerode(maskSky, strel('disk', 3) );
%maskG = imerode(maskSky, strel('disk', 3) );

%calculating the mean of the sky color
imageActual = imread([GeneralDataFolder '/' ImgFolder '/' filename '.jpg']);
meanSkyColor = permute( sum( sum( repmat(maskSky, [1 1 3]) .* ...
				double( imresize(imageActual, size(maskSky) ) ), 1), 2), [3 1 2]) ;
meanGroundColor = permute( sum( sum( repmat(maskG, [1 1 3]) .* ...
				double( imresize(imageActual, size(maskG) ) ), 1), 2), [3 1 2]) ;
meanSkyColor = meanSkyColor / sum(sum( maskSky ) );
meanGroundColor = meanGroundColor / sum(sum( maskG ));

vrml_filename = [filename '_' DepthDirectory '.wrl'];
%LowResImgIndexSuperpixelSepTemp = LowResImgIndexSuperpixelSep;
%[VertYSize HoriXSize] = size(LowResImgIndexSuperpixelSepTemp);
[dum VertYSize HoriXSize] = size(Position3D);
%nu_patch = VertYSize* HoriXSize;

Position3DCoord = reshape(Position3D,3,VertYSize*HoriXSize);    
Position3DCoord(3,:) = -Position3DCoord(3,:); % important to make z direction negative

table = reshape(0:(VertYSize*HoriXSize-1),VertYSize,[]);
Position3DCoordOrdeerIndex = zeros(1,92568); %RAJIV
GridPosition3DCoordOrdeerIndex = zeros(1,92576);
HBrokeBook(:,end) = 0;
VBrokeBook(end,:) = 0;

if displayFlag,
disp('Going into FOR LOOP, which is bad in Matlab. This code should not take longer than 3 seconds');
end

for i = 1:VertYSize-1
    for j = 1:HoriXSize-1
        % check which diagnoal to render
        if isempty(HBrokeBook) | isempty(VBrokeBook)
           Index = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i+1,j+1) table(i,j+1)];
           GridIndex = [table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i,j) table(i+1,j+1) table(i,j+1) table(i,j)];
        else 
           target = [HBrokeBook(i,j) VBrokeBook(i,j+1) HBrokeBook(i+1,j) VBrokeBook(i,j)];
           if target == [0 0 1 1];
              Index = [ table(i,j) table(i+1,j+1) table(i,j+1) table(i,j) table(i+1,j) table(i+1,j+1)];
              GridIndex = [table(i,j) table(i+1,j+1) table(i,j+1) table(i,j)];
           elseif target == [1 0 0 1];
              Index = [ table(i+1,j) table(i+1,j+1) table(i,j+1) table(i,j) table(i+1,j) table(i,j+1)];
              GridIndex = [table(i+1,j) table(i+1,j+1) table(i,j+1) table(i+1,j)];
           elseif target == [1 1 0 0];
              Index = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i+1,j+1) table(i,j+1)];
              GridIndex = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j)];
           elseif target == [0 1 1 0];
              Index = [ table(i,j) table(i+1,j) table(i,j+1) table(i+1,j) table(i+1,j+1) table(i,j+1)];
              GridIndex = [ table(i,j) table(i+1,j) table(i,j+1) table(i,j)];
           else 
              Index = [ table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i+1,j+1) table(i,j+1)];
              GridIndex = [table(i,j) table(i+1,j) table(i+1,j+1) table(i,j) table(i,j) table(i+1,j+1) table(i,j+1) table(i,j)];
           end
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
               Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index];
               GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex];
                %% also update the depth values for all the 6 points by
                %% changing them to a value of any non-sky point in the
                %% vicinity of Np down and to the right

                %% so first we find the point non-sky point which made us
                %% include this sky point
                iRep=i;
                indexVec=j:jL;
                nonSkyPoints=find(maskSky(i,indexVec)==0);
                if length(nonSkyPoints)>0
                    jRep=indexVec(a(1));
                else
                    jRep=j;
                    indexVec=i:iL;
                    nonSkyPoints=find(maskSky(indexVec,j)==0);
                    iRep=indexVec(nonSkyPoints(1));
                end
                %% now we update the points
                indexArray = [i i+1 i+1 i i+1 i; j j j+1 j j+1 j+1];

                for iA=1:6
                    %if indexArray(1,iA)>VertYSize || indexArray(2,iA)>HoriXSize
                    %   indexArray(2,iA)
                    %   indexArray(1,iA)
                    %end
                    if maskSky(indexArray(1,iA),indexArray(2,iA))==1
                        newPosition3D = permute(ray(indexArray(1,iA),indexArray(2,iA),:),[3 2 1])*depth(iRep,jRep);
                        Imgindex = sub2ind([VertYNuDepth HoriXNuDepth],indexArray(1,iA),indexArray(2,iA));
                        Position3Dcoord(:,Imgindex)=newPosition3D;
                    end
                end
            end
        else
            Position3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex Index];
            GridPosition3DCoordOrdeerIndex = [Position3DCoordOrdeerIndex GridIndex];
            %% also update the depth values
            indexArray = [i+1 i+1 i+1 i; j j+1 j+1 j+1];
            for iA=1:4
                if maskSky(indexArray(1,iA),indexArray(2,iA))==1
                    newPosition3D = permute(ray(indexArray(1,iA),indexArray(2,iA),:),[3 2 1])*depth(i,j);
                    Imgindex = sub2ind([VertYNuDepth HoriXNuDepth],indexArray(1,iA),indexArray(2,iA));
                    Position3Dcoord(:,Imgindex)=newPosition3D;
                end
            end
        end

        
    end
end
size(Position3DCoordOrdeerIndex)
size(GridPosition3DCoordOrdeerIndex)

% calculate PositionTexCoord
PositionTex = permute(ray(:,:,1:2)./repmat(cat(3,a,b),[VertYSize HoriXSize 1])+0.5,[3 1 2]);
PositionTexCoord = PositionTex;   

% find the Sky Wall point
Xmin=min(Position3DCoord(1,:))-Sep;
Xmax=max(Position3DCoord(1,:))+Sep;
Ymin=min(Position3DCoord(2,:))-Sep;
Ymax=max(Position3DCoord(2,:))+Sep;
Zmin=min(Position3DCoord(3,:))-Sep;
Zmax=max(Position3DCoord(3,:))+Sep;
%% the eight corners of the cube would then be
cc=[];
cc = [cc [Xmin Ymin Zmax]']; %% 1
cc = [cc [Xmax Ymin Zmax]']; %% 2
cc = [cc [Xmax Ymin Zmin]']; %% 3
cc = [cc [Xmin Ymin Zmin]']; %% 4

cc = [cc [Xmin Ymax Zmax]']; %% 5
cc = [cc [Xmax Ymax Zmax]']; %% 6
cc = [cc [Xmax Ymax Zmin]']; %% 7
cc = [cc [Xmin Ymax Zmin]']; %% 8
% wallIndices = [front_wall right_wall back_wall left_wall];
wallIndices = [2 6 7 3 3 7 8 4 1 4 8 5 5 8 7 6 1 2 3 4 1 5 6 2]-1;

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

%============== add background color======
fprintf(fp, 'DEF Back1 Background {\n');
%fprintf(fp, 'groundColor [.3 .29 .27]\n');
fprintf(fp, 'groundColor [%f %f %f]\n', meanGroundColor/255);
%fprintf(fp, 'skyColor [0.31 0.54 0.76]}\n');
fprintf(fp, 'skyColor [%f %f %f]}\n', meanSkyColor/255);
%=========================================

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

disp('Finished writing vrml');
