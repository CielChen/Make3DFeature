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
function [Proj] = LineProj(seglist, x , y)

% this function output the porject point on the given lineSeg

if isempty(x) && isempty(y)
   Proj = sparse(0,1);
   return;
end

if isempty(x)
   Proj = (seglist(1)-seglist(3))/(seglist(2)-seglist(4))*(y -seglist(2))+seglist(1);
   if any(isinf(Proj))
      disp('Proj_InfX');
   end
elseif isempty(y)
   Proj = (seglist(2)-seglist(4))/(seglist(1)-seglist(3))*(x -seglist(1))+seglist(2);
   if any(isinf(Proj))
      disp('Proj_InfY');
   end
else
   if seglist(1)-seglist(3) == 0
      Proj = x <= seglist(1);
   elseif seglist(2)-seglist(4) == 0
      Proj = y <= seglist(2);
   else   
%      Proj = (x - seglist(1))./(y - seglist(2)) - (seglist(1)-seglist(3))/(seglist(2)-seglist(4));
%       temp = (seglist(2)-seglist(4))/(seglist(1)-seglist(3))*(x -seglist(1))+seglist(2);
%       Proj = (y - temp)>=0;
     LineTarget = (seglist(:,1)-seglist(:,3) ~=0);
       temp = repmat((seglist(LineTarget,2)-seglist(LineTarget,4))./(seglist(LineTarget,1)-seglist(LineTarget,3)),[1 size(x,1)]).*...
              (repmat(x',[size(seglist(LineTarget,1),1) 1]) -repmat(seglist(LineTarget,1),[1 size(x,1)]))+repmat(seglist(LineTarget,2),[1 size(x,1)]);
       Proj = ((repmat(y',[size(seglist(LineTarget,1),1) 1]) - temp)>=0)';
     LineTarget = (seglist(:,1)-seglist(:,3) ==0);
       OProj = (( repmat(x',[size(seglist(LineTarget,1),1) 1]) - repmat(seglist(LineTarget,1),[1 size(x,1)]))>=0)';
     if sum(LineTarget(:))~=0
       Proj = [Proj OProj];
     end
   end
end
return;

