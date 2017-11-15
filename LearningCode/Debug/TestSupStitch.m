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

%load /afs/cs/group/reconstruction3d/scratch/testE/BeforeSupRay.mat
load /afs/cs/group/reconstruction3d/scratch/testE/PlaneParaPics2.mat
ImNu = 304;

SampleIndexYSmall = repmat((1:Default.VertYNuDepth)',[1 Default.HoriXNuDepth]);
  SampleIndexXSmall = repmat((1:Default.HoriXNuDepth),[Default.VertYNuDepth 1]);
  SampleImCoordYSmall = (( Default.VertYNuDepth+1-SampleIndexYSmall)-0.5)/Default.VertYNuDepth - Default.Oy_default;
  SampleImCoordXSmall = ( SampleIndexXSmall-0.5)/Default.HoriXNuDepth - Default.Ox_default;
  RayOri = RayImPosition( SampleImCoordYSmall, SampleImCoordXSmall,...
                          Default.a_default, Default.b_default, ...
                          Default.Ox_default,Default.Oy_default); %[ horiXSizeLowREs VertYSizeLowREs 3]
  RayOri = permute(RayOri,[3 1 2]); %[ 3 horiXSizeLowREs VertYSizeLowREs]

img = imresize(img, [ Default.SegVertYSize Default.SegHoriXSize]);
  [seglist]=edgeSegDetection(img,i,0);
  
     DisplaySup(MedSup,ImNu);
     hold on; drawseg(seglist,ImNu);
  
seglistSmall(:,1:2) = Matrix2ImgCo(Default.SegHoriXSize, Default.SegVertYSize, seglist(:,1:2)); 
  seglistSmall(:,3:4) = Matrix2ImgCo(Default.SegHoriXSize, Default.SegVertYSize, seglist(:,3:4)); 
  seglistSmall(:,1:2) = ImgCo2Matrix(Default.HoriXNuDepth, Default.VertYNuDepth, seglistSmall(:,1:2)); 
  seglistSmall(:,3:4) = ImgCo2Matrix(Default.HoriXNuDepth, Default.VertYNuDepth, seglistSmall(:,3:4)); 
VB = MedSup(:,round(linspace(1, Default.SegHoriXSize, Default.HoriXNuDepth)));
  HB = MedSup(round(linspace(1, Default.SegVertYSize, Default.VertYNuDepth)),:);
% ==========core

TextCoor = SupRayAlign( Sup{1}, VB, HB, seglistSmall, SampleIndexXSmall, SampleIndexYSmall, SupSize);
% ==============
%Test
TextCoorMed = Matrix2ImgCo( Default.HoriXNuDepth, Default.VertYNuDepth, reshape( permute( TextCoor, [3 1 2 ]),2,[])' );
TextCoorMed = ImgCo2Matrix( Default.SegHoriXSize, Default.SegVertYSize, TextCoorMed);
figure(ImNu); hold on;
scatter(TextCoorMed(:,1),TextCoorMed(:,2), 'd');
TextCoorOri = Matrix2ImgCo( Default.HoriXNuDepth, Default.VertYNuDepth, [SampleIndexXSmall(:) SampleIndexYSmall(:)]);
TextCoorOri = ImgCo2Matrix( Default.SegHoriXSize, Default.SegVertYSize, TextCoorOri);
figure(ImNu); hold on;
scatter(TextCoorOri(:,1),TextCoorOri(:,2), 0.5*ones(size(TextCoorOri(:,1))));

