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
function Default = SetupDefault_New(taskName, ParaFolder, OutPutFolder, ScratchFolder, Flag)
% This script setup the default parameter for the whole LearningCode group

Default.filename{1} = taskName; %'Train400' %'MinT';%'AshOldTrain';%'LineTest'%'MinDataOct21';%'MinT';
Default.ParaFolder = ParaFolder; 
Default.OutPutFolder = OutPutFolder; 
Default.ScratchFolder = ScratchFolder;

% setflag
if isempty( Flag)
   Default.Flag.DisplayFlag = 0;  % DisplayFlag=0£¬²»ÏÔÊ¾³¬ÏñËØ·Ö¸îµÄÍ¼Æ¬£»DisplayFlag=1£¬ÏÔÊ¾³¬ÏñËØ·Ö¸îµÄÍ¼Æ¬
   Default.Flag.IntermediateStorage = 0;
   Default.Flag.FeaturesOnly = 0;
   Default.Flag.NormalizeFlag = 1;
   Default.Flag.BeforeInferenceStorage = 0;
   Default.Flag.AfterInferenceStorage = 0
   Default.Flag.NonInference = 0;
else
   Default.Flag = Flag;
end

% ============== Highly changealbe parameters ========================
Default.SegVertYSize = 900;%1200;
Default.SegHoriXSize = 1200;%900;
Default.VertYNuPatch = 55;
Default.HoriXNuPatch = 61;%305;%61;
% Default.VertYNuDepth = 55;
% Default.HoriXNuDepth = 305;
Default.VertYNuDepth = 165;  %³¬ÏñËØµÄ´óÐ¡
Default.HoriXNuDepth = 915;
Default.PopUpHoriX = 800;%600;
Default.PopUpVertY = 600;%800;
Default.batchSize = 10;
Default.NuRow_default = 55;
Default.WeiBatchSize = 5;
Default.TrainVerYSize = 1704;%2272;
Default.TrainHoriXSize = 2272;%1704;
Default.MempryFactor =2;
% pics info
Default.Horizon = 1/2;% the position of the horizon in a pics (the bottom of the pic is 0 top is 1 middle is 1/2)
% segmentation info
Default.sigm = 0.5;%0.8%0.3;
Default.k = 100;%300;%200;
Default.minp = 100;%150;%20;
% Superpixel cleaning parameter
Default.SmallThre = 5; %smallest sup size (used in Features/premergAllsuperpixel_efficient.m only)
% ====================================================================

% =============
% Default.PpmOption = 1; % option to storage the ppm image segmentation
Default.PpmOption = 0;
% ============

%================ camera info from kyle's code
% This can probably also be estimated from jpeg header
Default.fy = 2400.2091651084;
Default.fx = 2407.3312729885838;
Default.Ox = 1110.7122391785729;%2272/2; %
Default.Oy = 833.72104535435108;%1704/2; %
Default.a_default = 2272/Default.fx; %0.70783777; %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
Default.b_default = 1704/Default.fy; %0.946584169;%0.085; % vertical physical size of image plane normalized to focal length (in meter)
Default.Ox_default = 1-Default.Ox/2272;%0.489272914; % camera origin offset from the image center in horizontal direction
Default.Oy_default = 1-Default.Oy/1704;%0.488886982; % camera origin offset from the image center in vertical direction

Default.GroundThreshold = 0.5;
Default.SkyThreshold = 1;

return;
