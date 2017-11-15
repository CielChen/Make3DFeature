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
function Gen1StepInferenceData(Fdir, ImgName)

%OutPutFolder = '/afs/cs/group/reconstruction3d/scratch/3DmodelMultipleImage/';
OutPutFolder = [[Fdir '/Wrl/' ImgName '/']];
system(['mkdir ' OutPutFolder]);

% Test if .wrl model exist already
%[ status, result ] = system([ 'ls ' OutPutFolder '*.wrl'] );
%if ~status
%	disp('Wrl already exist');
%	return;
%end

%OutPutFolder = '/afs/cs/group/reconstruction3d/scratch/Popup/result/';
taskName = '';%(Not Used) taskname will append to the imagename and form the outputname
Flag.DisplayFlag = 0;
Flag.IntermediateStorage = 0;
Flag.FeaturesOnly = 0;
Flag.FeatureStorage = 0; %Min add May 2nd
Flag.NormalizeFlag = 1;
Flag.BeforeInferenceStorage = 1;
Flag.NonInference = 0;
Flag.AfterInferenceStorage = 1;
%ScratchFolder = '/afs/cs/group/reconstruction3d/scratch/'; % ScratchFolder
%ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/'
%AImgPath = '/afs/cs.stanford.edu/group/reconstruction3d/Data/building0010.jpg'
%ScratchFolder = '/afs/cs/group/reconstruction3d/scratch/temp'; % ScratchFolder
ScratchFolder = [Fdir '/data/' ImgName]%IMG_0614'; % ScratchFolder
system(['mkdir ' ScratchFolder]);
ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/'
%AImgPath = '/afs/cs.stanford.edu/group/reconstruction3d/scratch/Popup/images/VenetianLasVegas.jpg'
AImgPath = [Fdir '/jpg/' ImgName '.jpg']

      OneShot3dEfficient(AImgPath, OutPutFolder,...
        taskName,...% taskname will append to the imagename and form the outputname
        ScratchFolder,... % ScratchFolder
        ParaFolder,...
        Flag...  % All Flags 1) intermediate storage flag
        );
