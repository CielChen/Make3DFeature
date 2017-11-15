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
function Default=SetupDefault(tskName,imgFolder,trainSet,learnType,learnSkyEx,learnLog,learnNear,learnAlg,learnDate,absFeaType, ...
            absFeaDate,histFeaType,histFeaDate,generalDataFolder,scratchDataFolder,localFolder,clusterExecutionDir)
% This script setup the default parameter for the whole LearningCode group

    Default.taskName = tskName; %'Train400' %'MinT';%'AshOldTrain';%'LineTest'%'MinDataOct21';%'MinT';
    Default.ImgFolder = imgFolder; %'Train400'%'ProperImg'; %'Test134';%'AshOldData343';%'LineProImg' %'SUImg'%'Dataset_Oct21';%'ProperImg';
    Default.TrainSet = trainSet; %'Train400'%'ProperImg';
    Default.LearnType = learnType; %'Abs'%'Fractional';
    Default.LearnSkyEx = learnSkyEx; %1;
    Default.LearnLog = learnLog;%0;
    Default.LearnNear = learnNear; %0;
    Default.LearnAlg = learnAlg; %'robustfit'
    Default.LearnDate = learnDate; %'';
    Default.AbsFeaType = absFeaType; %'Whole';
    Default.AbsFeaDate = absFeaDate; %'';
    Default.HistFeaType = histFeaType; %'None';
    Default.HistFeaDate = histFeaDate; %'';

% define general data folder and scratch data folder

Default.GeneralDataFolder = generalDataFolder; %'/afs/cs/group/reconstruction3d/Data';
Default.ScratchDataFolder = scratchDataFolder; %['/afs/cs/group/reconstruction3d/scratch/' Default.taskName]
Default.LocalFolder = localFolder; %pwd;
Default.ClusterExecutionDirectory = clusterExecutionDir; %'./ClusterExecuationDirectory';
% ============== Highly changealbe parameters ========================
Default.SegVertYSize = 1200;
Default.SegHoriXSize = 900;
Default.VertYNuPatch = 55;
Default.HoriXNuPatch = 61;%305;%61;
Default.VertYNuDepth = 55;
Default.HoriXNuDepth = 305;
Default.PopUpHoriX = 600;
Default.PopUpVertY = 800;
Default.batchSize = 10;
Default.NuRow_default = 55;
Default.WeiBatchSize = 5;
Default.TrainVerYSize = 2272;
Default.TrainHoriXSize = 1704;
Default.MempryFactor =2;
% pics info
Default.Horizon = 1/2;% the position of the horizon in a pics (the bottom of the pic is 0 top is 1 middle is 1/2)
% segmentation info
Default.sigm = 0.5;%0.8%0.3;
Default.k = 100;%300;%200;
Default.min = 100;%150;%20;
% ====================================================================



%================ camera info from kyle's code
% This can probably also be estimated from jpeg header
Default.fx = 2400.2091651084;
Default.fy = 7.3312729885838;
Default.Oy = 1110.7122391785729;%2272/2; %
Default.Ox = 833.72104535435108;%1704/2; %
Default.a = 1704/Default.fy; %0.70783777; %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
Default.b = 2272/Default.fx; %0.946584169;%0.085; % vertical physical size of image plane normalized to focal length (in meter)
Default.Ox = 1-Default.Ox/1704;%0.489272914; % camera origin offset from the image center in horizontal direction
Default.Oy = 1-Default.Oy/2272;%0.488886982; % camera origin offset from the image center in vertical direction

