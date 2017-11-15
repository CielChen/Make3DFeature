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
%function [] =RenderRelations(Fdir)

file = dir([ Fdir '/jpg/']);
NuItem = size(file,1);
j = 0;
for i = 1:NuItem
	if size(strfind(file(i).name,'.jpg'),1) ~= 0
		j = j + 1;
		filename{j} = strrep(file(i).name,'.jpg','');

		DataFolder = [Fdir '/data/' filename{j}];
		load([DataFolder '/' filename{j} '__AInfnew.mat']);
		WeiM = WeiM + WeiM';

		% detect boundary
		BounaryPHori = conv2(Sup,[1 -1],'same') ~=0;
		BounaryPHori(:,end) = 0;
		BounaryPVert = conv2(Sup,[1; -1],'same') ~=0;
		BounaryPVert(end,:) = 0;
		
		ReL = ones(size(Sup));
		SupHoriRight = 	Sup(logical( BounaryPHori));
		Temp = [ zeros(size(Sup,1) ,1) BounaryPHori(:,1:(end-1)) ];
		SupHoriLeft = Sup(logical( Temp ));
		EraseRowMark = SupHoriRight == 0 | SupHoriLeft == 0;
		SupHoriRight(EraseRowMark) = [];
		SupHoriLeft(EraseRowMark) = [];

		IndHori = sub2ind(size(WeiM), SupHoriRight, SupHoriLeft	);
		ReL(BounaryPHori) = WeiM(IndHori);

		SupVertTop = 	Sup(logical( BounaryPVert));
		Temp = [ zeros(1,size(Sup,2) ); BounaryPVert(1:(end-1),:) ];
		SupVertBottom = Sup(logical( Temp ));
		EraseRowMark = SupVertTop == 0 | SupVertBottom == 0;
		SupVertTop(EraseRowMark) = [];
		SupVertBottom(EraseRowMark) = [];

		IndVert = sub2ind(size(WeiM), SupVertTop, SupVertBottom	);
		ReL(BounaryPVert) = WeiM(IndVert);
		ReLM{j} = ReL;
		Nu{j} = 1./exp(abs(VarMapRaw));
	end;

	if j == 1
%		break;
	end
end

return;
