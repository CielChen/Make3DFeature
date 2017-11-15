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
%ImgFolder = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/COM5_070609_221936_gate_hilbert_more';
%ImgFolder{1} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/McCosh';
%ImgFolder{2} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/Joline';
%ImgFolder{3} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/StudioCenter';
%ImgFolder{4} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/GilbertPanarama';
%ImgFolder{1} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/1215';
%ImgFolder{1} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/Lun_de_Miel';
%ImgFolder{1} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/QuadNew';
ImgFolder{1} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/QuadCorner';
%ImgFolder{3} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/OldUnion';
%ImgFolder{4} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/Dorm';
%ImgFolder{5} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/Gym';
%ImgFolder{6} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/Maples';
%ImgFolder{7} = '/afs/cs/group/reconstruction3d/scratch/TestMultipleImage/SunkenDimoad';

for k = 1:length(ImgFolder)
	file = dir([ImgFolder{k}  '/jpg/']);
	j = 0;
	NuItem = size(file,1);
	system(['mkdir ' ImgFolder{k} '/data']);
	system(['mkdir ' ImgFolder{k} '/Wrl']);        
	[status, result] = system([ 'ls ' ImgFolder{k} '/Wrl/*/*_.wrl']);
	EndStr = strfind(result,'.wrl');
	StartStr = strfind(result,'/afs/');
	fileWrl = [];
	for q = 1:length(StartStr)
		fileWrl(q).name = result(StartStr(q):EndStr(q)+3);
	end

	for i = 1:NuItem
	    GenWrl = false;
	    if size(strfind(file(i).name,'.jpg'),1) ~= 0
	        j = j + 1;
        	filename{j} = strrep(file(i).name,'.jpg','');
	    elseif size(strfind(file(i).name,'.JPG'),1) ~= 0
        	j = j + 1;
        	filename{j} = strrep(file(i).name,'.JPG','');
        	% change the .JPG to .jpg
	    else
		continue;
    	    end

	    l = 1;
	    while ~GenWrl & l <= length(fileWrl)
  		GenWrl = strcmp(fileWrl(l).name, [ ImgFolder{k} '/Wrl/' filename{j} '/' filename{j} '_.wrl']);
		l = l + 1;
	    end

	    if ~GenWrl
		filename{j}
        	Gen1StepInferenceData(ImgFolder{k}, filename{j});
	    end
	end
end

