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
function [Sup] = FillIndex(Sup)

% this function fill the holes in Sup == 0
% by Active the ActiveMask area

Numholes = sum( Sup(:) == 0)
[TargetHolesY, TargetHolesX] = find( Sup ==0);
ScaleSup = size( Sup);
ActiveMask = Sup > 0;
while Numholes >0
	
	for i = 1:Numholes
		RegionOfInterest = Sup( ...
					max( min( (TargetHolesY(i)-1), ScaleSup(1)), 1):max( min( (TargetHolesY(i)+1),ScaleSup(1)), 1), ...
				        max( min( (TargetHolesX(i)-1), ScaleSup(2)), 1):max( min( (TargetHolesX(i)+1),ScaleSup(2)), 1) ) ...
					.*ActiveMask( ...
					max( min( (TargetHolesY(i)-1), ScaleSup(1)), 1):max( min( (TargetHolesY(i)+1),ScaleSup(1)), 1), ...
				        max( min( (TargetHolesX(i)-1), ScaleSup(2)), 1):max( min( (TargetHolesX(i)+1),ScaleSup(2)), 1) );
		Element = RegionOfInterest(:);%unique( RegionOfInterest);
		% Remove zeros and negative or NaN element
% 		Element = setdiff(Element, NaN);
        Element = Element( Element ~=NaN);
		Element = Element( Element >0);
		if isempty( Element)
			continue;
		end
		Sup( TargetHolesY(i), TargetHolesX(i)) = mode( Element(:));
	end

	% check if all the holes filled	
	Numholes = sum( Sup(:) == 0)
	[TargetHolesY TargetHolesX] = find( Sup ==0);
    ActiveMask = Sup > 0;
end

return;
