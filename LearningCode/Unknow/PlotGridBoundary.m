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
function [] = PlotGridBoundary( BoundaryHori, BoundaryVert, VertexX, VertexY, Res, h, Color)

% this function plot the Bundary in the specifited Resolution
[yr xr] = size(VertexX);
Vertex = ImgCo2Matrix(Res(2), Res(1), [VertexX(:) VertexY(:)]); % now in the resolution of HoriXNuDepth, VertYNuDepth
VertexY = reshape(Vertex(:,2), yr, []);
VertexX = reshape(Vertex(:,1), yr, []);
[yH xH] = find( BoundaryHori ~=0);
IndH = sub2ind([yr xr], yH, xH);
[yV xV] = find( BoundaryVert ~=0);
IndV = sub2ind([yr xr], yV, xV);
X = [[VertexX(IndH+1)'; VertexX(IndH+1+yr)']  [VertexX(IndV+yr)'; VertexX(IndV+1+yr)']];
Y = [[VertexY(IndH+1)'; VertexY(IndH+1+yr)']  [VertexY(IndV+yr)'; VertexY(IndV+1+yr)']];
figure(h);
line(X,Y,'Color',Color);

return;
