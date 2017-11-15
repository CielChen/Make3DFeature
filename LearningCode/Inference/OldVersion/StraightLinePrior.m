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
function [StraightLinePriorSelectMatrix, RayPorjectImgMapY,RayPorjectImgMapX, MovedPatchBook, LineIndex,HBrokeBook,VBrokeBook] = ...
          StraightLinePrior(p,Img,lineSegList,LearnedDepth,RayPorjectImgMapY,RayPorjectImgMapX,MovedPatchBook,HBrokeBook,VBrokeBook,Verti,a,b,Ox,Oy);

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;


        
        % initialize output
        LineIndex = [];
        StraightLinePriorSelectMatrix = [] ;

        % take out the line seg whiich is too short (2 can be changed)
        if Verti == 0
      	   lineSegList(abs(max(min(round(lineSegList(:,1)),HoriXNuDepth),1)...
                    -max(min(round(lineSegList(:,3)),HoriXNuDepth),1))<=2,:)=[];
           figure(200); imagesc(Img); hold on;
        else
      	   lineSegList(abs(max(min(round(lineSegList(:,2)),VertYNuDepth),1)...
                    -max(min(round(lineSegList(:,4)),VertYNuDepth),1))<=2,:)=[];
           figure(300); imagesc(Img); hold on;

        end
        
        [ImgYSize ImgXSize dummy] = size(Img);
        NuSeg = size(lineSegList,1);

        % store the line
        tempLine = ((lineSegList+0.5)./repmat([HoriXNuDepth VertYNuDepth HoriXNuDepth VertYNuDepth],NuSeg,1)...
                    .*repmat([ImgXSize ImgYSize ImgXSize ImgYSize],NuSeg,1))-0.5;
        if Verti==0
           drawseg(tempLine,200,2);
           saveas(200,[ScratchDataFolder '/LineSeg/' filename{p} '_lineSegHori.jpg']);
        else
           drawseg(tempLine,300,2);
           saveas(300,[ScratchDataFolder '/LineSeg/' filename{p} '_lineSegVert.jpg']);
        end

        if NuSeg == 0
           LineIndex{1,1} = [];
           LineIndex{1,2} = [];
           LineIndex{1,3} = [];
           LineIndex{1,4} = [];
           StraightLinePriorSelectMatrix = sparse(1,VertYNuDepth*HoriXNuDepth);
           disp('NuSeg==0')
           return;
        end

        % Sort by segemtn length
        List = sortrows([norms((lineSegList(:,1:2)-lineSegList(:,3:4))')' (1:NuSeg)'],1);
        lineSegList = lineSegList(List(:,end),:);

        % stick the patch to the straight line(If don't output ayPorjectImgMapY then no stick)
        %RayPorjectImgMapY = repmat((1:VertYNuDepth)',1,HoriXNuDepth);
        Count = 1
        for k = NuSeg:-1:(1)
            % first find fixed x point to flexible y point
            if lineSegList(k,1)>lineSegList(k,3) % line always start from left to right
                Start = lineSegList(k,[3 4]);
                End = lineSegList(k,[1 2]);
            else
                End = lineSegList(k,[3 4]);
                Start = lineSegList(k,[1 2]);
            end

            % find out all the close point to the line
            X = max(ceil(Start(1)),1):min(floor(End(1)),HoriXNuDepth); %Define the fixed Horizontal position
            Y = Start(2)+(End(2)-Start(2))/(End(1)-Start(1)).*(X - Start(1)); %Corresponded Vertical position
            % make sure no Y <=1 and Y >= VertYNuDepth
            mask = Y <= 1 | Y >= VertYNuDepth;
            X(mask) = [];
            Y(mask) = [];
 
            % keep the record of the index that been stick to the line
            LineIndex{Count,1} =sub2ind([VertYNuDepth HoriXNuDepth ],min(max(floor(Y),1),VertYNuDepth),X);
            VBrokeBook(setdiff(LineIndex{Count,1},...
                sub2ind([VertYNuDepth HoriXNuDepth ],VertYNuDepth*ones(HoriXNuDepth,1),(1:HoriXNuDepth)'))) = 1;
            % make sure no LineIndex{Count,1} == LineIndex{Count,2}
            temp =sub2ind([VertYNuDepth HoriXNuDepth ],max(min(ceil(Y),VertYNuDepth),1),X);
            temp(temp==LineIndex{Count,1}) = temp(temp==LineIndex{Count,1})+1;
            LineIndex{Count,2} = temp;

            % first find fixed y point to flexible x point
            if lineSegList(k,2)>lineSegList(k,4) % line always start from top to bottom
                Start = lineSegList(k,[3 4]);
                End = lineSegList(k,[1 2]);
            else
                End = lineSegList(k,[3 4]);
                Start = lineSegList(k,[1 2]);
            end

            VY = max(ceil(Start(2)),1):min(floor(End(2)),VertYNuDepth); %Define the fixed Horizontal position
            VX = Start(1)+(End(1)-Start(1))/(End(2)-Start(2)).*(VY - Start(2)); %Corresponded Vertical position
            % make sure no Y <=1 and Y >= VertYNuDepth
            mask = VX <= 1 | VX >= HoriXNuDepth;
            VX(mask) = [];
            VY(mask) = [];

            % keep the record of the index that been stick to the line
            LineIndex{Count,3} =sub2ind([VertYNuDepth HoriXNuDepth ],VY,min(max(floor(VX),1),HoriXNuDepth));
            HBrokeBook(setdiff(LineIndex{Count,3},...
                sub2ind([VertYNuDepth HoriXNuDepth ],(1:VertYNuDepth)',HoriXNuDepth*ones(VertYNuDepth,1)))) = 1;
            % make sure no LineIndex{Count,1} == LineIndex{Count,2}
            temp = sub2ind([VertYNuDepth HoriXNuDepth ],VY,max(min(ceil(VX),HoriXNuDepth),1));
            temp(temp==LineIndex{Count,3}) = temp(temp==LineIndex{Count,3})+1;
            LineIndex{Count,4} = temp;

            if Verti == 1
              X = VX;
              Y = VY; 
              IndexUpper = LineIndex{Count,3};
              IndexLower = LineIndex{Count,4};
            else
              IndexUpper = LineIndex{Count,1};
              IndexLower = LineIndex{Count,2};
            end

            % make sure X is not empty
            if size(X,2) == 0
               continue;
            end
 
            % generate specific ray for whole line
            RayLineY = ((VertYNuDepth+1-Y)-0.5)/VertYNuDepth - Oy;
            RayLineX = ((X)-0.5)/HoriXNuDepth - Ox;
            RayCenter = RayImPosition(RayLineY,RayLineX,a,b,Ox,Oy); %[ horiXSizeLowREs VertYSizeLowREs 3]
            % generating the axis of the plane that these rays lay on
            [U S V] = svd(permute(RayCenter,[3 2 1]));
            Ns = U(:,3);
            U = U(:,1:2);
            S = S(1:2,1:2);
            V = V(:,1:2);

            % calculate all the points in this plane coordinate U

            LSize = size(IndexUpper,2);
            PlaneCoordUpper = S*V'*[spdiags(LearnedDepth(IndexUpper)',0,LSize,LSize)]; 
            PlaneCoordLower = S*V'*[spdiags(LearnedDepth(IndexLower)',0,LSize,LSize)]; 
         
            if 0 %Sep(k)==1 
               % fit line for all the points in IndexUpper and IndexLower
               cvx_begin
                  cvx_quiet(true);
                  variable m(1);
                  variable c(1);
                  minimize(norm([-PlaneCoordUpper(2,:)' -ones(LSize,1)]*[m;c]+PlaneCoordUpper(1,:)',1));
               cvx_end

               % generating NL vector
               NL = cross(Ns,U(:,2)+m*U(:,1));
               NLUpper = NL./norm(NL);

               % fit line for all the points in IndexUpper and IndexLower
              cvx_begin
                  cvx_quiet(true);
                  variable m(1);
                  variable c(1);
                  minimize(norm([-PlaneCoordLower(2,:)' -ones(LSize,1)]*[m;c]+PlaneCoordLower(1,:)',1));
               cvx_end

               % generating NL vector
               NL = cross(Ns,U(:,2)+m*U(:,1));
               NLLower = NL./norm(NL);
            else
               % fit line for all the points in IndexUpper and IndexLower
               cvx_begin
                  cvx_quiet(true);
                  variable m(1);
                  variable c(1);
                  minimize(norm([[-PlaneCoordUpper(2,:)';-PlaneCoordLower(2,:)'] -ones(LSize*2,1)]*...
                     [m;c]+[PlaneCoordUpper(1,:)';PlaneCoordLower(1,:)'],1));
               cvx_end
               % generating NL vector
               NL = cross(Ns,U(:,2)+m*U(:,1));
               NLLower = NL./norm(NL);
               NLUpper = NLLower;
            end

            [c,i] = setdiff(IndexUpper,MovedPatchBook);
            if Verti == 0
               RayPorjectImgMapY(c) = Y(i);
            else
               RayPorjectImgMapX(c) = X(i);
            end
            for l = 1:(LSize-1)
                if any(MovedPatchBook==IndexUpper(l)) ||any(MovedPatchBook==IndexUpper(l+1))
                   continue;
                end
                SelectMatrix = sparse(1,VertYNuDepth*HoriXNuDepth);
                if Verti == 1
                   if abs(NLUpper'*[0; 1; 0]) > abs(NLUpper'*[0; 0; 1])
                      NLUpper = [0; 1; 0];
                   else
                      NLUpper = [0; 0; 1];
                   end
                end
                SelectMatrix(IndexUpper(l)) = NLUpper'*permute(RayCenter(1,l,:),[3 2 1]);
                SelectMatrix(IndexUpper(l+1)) = -NLUpper'*permute(RayCenter(1,l+1,:),[3 2 1]);
                StraightLinePriorSelectMatrix = [StraightLinePriorSelectMatrix; SelectMatrix];
            end
            MovedPatchBook = [MovedPatchBook c];
            % Lower part
            [c,i] = setdiff(IndexLower,MovedPatchBook);
            if Verti == 0
               RayPorjectImgMapY(c) = Y(i);
            else
               RayPorjectImgMapX(c) = X(i);
            end
            for l = 1:(LSize-1)
                if any(MovedPatchBook==IndexLower(l)) ||any(MovedPatchBook==IndexLower(l+1))
                   continue;
                end
                SelectMatrix = sparse(1,VertYNuDepth*HoriXNuDepth);
                if Verti == 1
                   if abs(NLLower'*[0; 1; 0]) > abs(NLLower'*[0; 0; 1])
                      NLLower = [0; 1; 0];
                   else
                      NLLower = [0; 0; 1];
                   end
                end
                SelectMatrix(IndexLower(l)) = NLLower'*permute(RayCenter(1,l,:),[3 2 1]);
                SelectMatrix(IndexLower(l+1)) = -NLLower'*permute(RayCenter(1,l+1,:),[3 2 1]);
                StraightLinePriorSelectMatrix = [StraightLinePriorSelectMatrix; SelectMatrix];
            end
            MovedPatchBook = [MovedPatchBook c];
            Count = Count +1;
        end
return; 
