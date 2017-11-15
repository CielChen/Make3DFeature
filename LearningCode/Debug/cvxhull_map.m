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
function cvx_sets = cvxhull_map(plane_pixels, grid_size)
% load plane_pixels.mat
% function get_convex_set(boundary_pixels)

% boundary_pixels are clockwise
% cvx_set = get_convex_set (boundary_pixels);

% we get the minimum y value and start from it
% as we move along the boundary we

% does not do anything, but later on, if we want to change the superpixels into convex shapes.
% Might solve the problem by rendering same parts twice in the wrl.

% The way to call: cvx_sets = cvxhull_map(plane_pixels, 10);
% for kk=1:length(cvx_sets)
%        boundary_pixels = [cvx_sets{kk}.x,cvx_sets{kk}.y]';

if ~exist('grid_size')
    grid_size=4;    % grid size
end

gsize= grid_size;
[m,n] = size(plane_pixels);

clear cvx_sets;
cvx_sets_count=0;

hold on;
for i=1:ceil((m-1)/gsize)
    for j=1:ceil((n-1)/gsize)
        patch = plane_pixels(gsize*(i-1)+1:min(gsize*i+1,m), gsize*(j-1)+1:min(gsize*j+1,n));
        [I J]= find(patch>0);
        I= I+ gsize*(i-1);
        J= J+ gsize*(j-1);
        if (length(I)==0) continue; end;
        
        try
        cvxh = convhull(J,I);
        catch
            a=1;
            continue;
        end
        
%         plot (J(cvxh), I(cvxh));
        cvx_sets_count= cvx_sets_count+1;
        cvx_sets{cvx_sets_count}.x = J(cvxh);
        cvx_sets{cvx_sets_count}.y = I(cvxh);
    end
end
hold off

