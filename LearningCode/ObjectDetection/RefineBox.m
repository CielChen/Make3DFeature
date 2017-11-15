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
function [NewBox] = RefineBox(Sup, Box, InScribeRatioThre);

NumBox = size(Box,1);
VOri = Box(:,3) - Box(:,1) + 1;
HOri = Box(:,4) - Box(:,2) + 1;
AreaOri = VOri.*HOri;
NewBox = [zeros(size(Box,1),4)];
OriShapeFlag = 0;
ShiftFlag = 1;
NumScale = 100;
for i = 1:NumBox
	
	BoxMask = zeros(size(Sup));
	BoxMask(Box(i,1):Box(i,3), Box(i,2):Box(i,4)) = 1;
	BoxMask = logical( BoxMask);
	ElementInscribe = setdiff( sort( unique( Sup( BoxMask)) ), 0);
	ElementOutSide = setdiff( sort( unique( Sup( ~BoxMask)) ), 0);
	ElementInscribeHistC = sparse(1, max(Sup(:)));
	ElementOutSideHistC = ElementInscribeHistC;
	ElementInscribeHistC( ElementInscribe) = histc(Sup( BoxMask),ElementInscribe);
	ElementOutSideHistC( ElementOutSide) = histc(Sup( ~BoxMask),ElementOutSide);
	InscribeRatio = ElementInscribeHistC./( ElementInscribeHistC+ElementOutSideHistC);
	InscribeRatio( isnan( InscribeRatio)) = 0;
	SupIndRefined = find( InscribeRatio > InScribeRatioThre);
	SupShape = zeros(size( Sup));
	for j = SupIndRefined
		SupShape( Sup == j) = InscribeRatio(j);
	end
	
    % decide new Box size
    SupShape(~BoxMask) = 0;
    Area = sum( SupShape(:) ~=0)
    Ratio = Area./AreaOri(i);

    % pick the Shape of the newBox
    if ~OriShapeFlag
	Vmax = VOri(i,1);
	Hmax = HOri(i,1);
	Vmin = Area/Hmax;
	count = 1;
	for Vtrial = linspace(Vmin,Vmax,NumScale)
	    V(count) = max( round( Vtrial), 1);	
	    H(count) = max( round( Area/Vtrial), 1);	
	    if ShiftFlag
		    h = ones(V(count),H(count));
		    res = filter2(h,SupShape(Box(i,1):Box(i,3), Box(i,2):Box(i,4)), 'valid');
		    [CCol ICol ] = max(res, [], 2);
		    [C IRow ] = max(CCol);
		    Shift(count,:) = [IRow ICol(IRow)] - 1;
		    Score(count) = C;	    
	    else
		Shift(count,:) = [VOri(i) HOri(i)] - [V(count) H(count)]; 
		Shift(count,:) = round(Shift(count,:)/2) - 1;
		Score(count) = sum( sum( SupShape( (Box(i,1):(Box(i,1)+V(count)))+Shift(count,1), (Box(i,2):(Box(i,2)+H(count)))+Shift(count,2))));
	    end
	    count = count + 1;
	end
	[BestScore BestIndex] = max(Score);
	NewBox(i,:) = [Box(i,1)+Shift(BestIndex,1) Box(i,2)+Shift(BestIndex,2) Box(i,1)+Shift(BestIndex,1)+V(BestIndex) Box(i,2)+Shift(BestIndex,2)+H(BestIndex)];
    else
	    V = max( round( VOri(i)*Ratio), 1);
	    H = max( round( HOri(i)*Ratio), 1);
            Shift = [VOri(i) HOri(i)] - [V H]; 
            Shift = round(Shift/2) - 1;
            NewBox(i,:) = Box(i,:) + [ Shift -Shift];
    end    
end
