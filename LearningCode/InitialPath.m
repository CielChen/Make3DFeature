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
function []=InitialPath(CompileFlag)

% 0) clean bin
%   system('rm -rf ./bin');

if nargin < 1
	CompileFlag = false;
end

if CompileFlag
% makefile and create every binary file in bin folder
  system('mkdir ./bin/');

  % InitialFolder = pwd;
  cd ./bin

% 1) ../third_party/Superpixels/SourceCode/segment/segmentImgOpt.cpp
  mex ../../third_party/Superpixels/SourceCode/segment/segmentImgOpt.cpp
%   system('mv ../third_party/Superpixels/SourceCode/segment/segmentImgOpt.mex* ./bin');

% 2) ./Debug/SparseAverageSample2DOptimized.cpp
  mex .././Features/SparseAverageSample2DOptimized.cpp
%   system('mv ./Debug/SparseAverageSample2DOptimized.mex* ./bin');

% 3) ./Debug/WrlFacestHroiReduce.cpp
  mex .././Rendering/WrlFacestHroiReduce.cpp
%   system('mv ./Debug/WrlFacestHroiReduce.mex* ./bin');

% 4) ./Debug/SupRayAlign.cpp
  mex .././Inference/SupRayAlign.cpp
%   system('mv ./Debug/SupRayAlign.mex* ./bin');

  cd ../;
end

% this script addd all path needed as flag s sepecified

% setup flag
CvxFlag = false;
YalmipFlag = true;
SeDuMiFlag = true;
LMIrankFlag = true;

%addpath(genpath('~/SVN_REPOSITORY/trunk/'));
addpath(genpath('../LearningCode/'));
addpath(genpath('../LaserDataCollection/'));
addpath(genpath('../third_party/EdgeLinkLineSegFit/'));
addpath(genpath('../ec2/bin/mex/')); 

if CvxFlag
% enble cvx
%   addpath('/afs/cs/group/reconstruction3d/Data/cvx');
   addpath('../third_party/opt/cvx');
   cvx_setup;
end

if YalmipFlag
   % add path for yalmip
%   addpath(genpath('/afs/cs/group/reconstruction3d/Data/yalmip'));
   addpath(genpath('../third_party/opt/yalmip'));
end

if SeDuMiFlag
   % add sedumi
%   path(path,'/afs/cs/group/reconstruction3d/Data/SeDuMi_1_1R3/SeDuMi_1_1');
   addpath(genpath('../third_party/opt/SeDuMi_1_1R3') );
end

if LMIrankFlag
   % add sedumi
%   path(path,'/afs/cs/group/reconstruction3d/Data/SeDuMi_1_1R3/SeDuMi_1_1');
   path(path,'../third_party/opt/lmirank');
end
