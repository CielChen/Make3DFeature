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
function [countPoints, Position3DCoord, Index, GridIndex]=removeOffsetPoints( TriPoints, Thresh, num, max_or_second_max, ray, indexArray, depth, ...
                    VertYNuDepth, HoriXNuDepth, Position3DCoord, Index, GridIndex, countPoints)
                
%% check the maximum or second maximum
switch (max_or_second_max)
    case 1 %% max        
		if (max( TriPoints)-min( TriPoints))>Thresh
            %% we bring the points forward and append to
            %% Position3DCoord
            [a1,b1]=min( TriPoints);
            [a2,b2]=max( TriPoints);
            newPosition3D = permute(ray(indexArray(1,(num-1)*3+b2),indexArray(2,(num-1)*3+b2),:),[3 2 1])*depth(indexArray(1,(num-1)*3+b1),indexArray(2,(num-1)*3+b1));
            Position3DCoord(:,countPoints)=newPosition3D;                   
            Position3DCoord(3,countPoints) = -Position3DCoord(3,countPoints); 
            Index((num-1)*3+b2)=countPoints;
            GridIndex((num-1)*4+b2)=countPoints;
            if b2==1
                GridIndex(4*num)=countPoints;
            end
            countPoints=countPoints+1;
		end     
    case 2 %% second max
        sortedVec=sort(TriPoints);
        if (sortedVec(2)-sortedVec(1))>Thresh
            %% we bring the points forward and append to
            %% Position3DCoord
            [a1,b1]=min( TriPoints);
            [a2,b2]=max( TriPoints);
            for loopVar=1:3
                if (loopVar~=b1)&&(loopVar~=b2)
                    b3=loopVar;
                end
            end
            newPosition3D = permute(ray(indexArray(1,(num-1)*3+b3),indexArray(2,(num-1)*3+b3),:),[3 2 1])*depth(indexArray(1,(num-1)*3+b1),indexArray(2,(num-1)*3+b1));
            Position3DCoord(:,countPoints)=newPosition3D;                    
            Position3DCoord(3,countPoints) = -Position3DCoord(3,countPoints); 
            Index((num-1)*3+b3)=countPoints;
            GridIndex((num-1)*4+b3)=countPoints;
            if b3==1
                GridIndex(4*num)=countPoints;
            end
            countPoints=countPoints+1;
		end
end       
