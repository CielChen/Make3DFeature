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
function [ maskg, maskSky]=gen_predicted_GS_efficient(Default, f, FSup)
%                           GroundThreshold, SkyThreshold)
%function [maskGvec, maskSkyvec]=gen_predicted_GS_efficient(TrainSet,HistFeaType,HistFeaDate,AbsFeaType,AbsFeaDate)

% This function generated the learned Ground and Sky mask (Code by Rajiv, Modified by Min 1/27)
% Input --
% f : (55x305=16775)by 104
% FSup: 13 by No. of Sup
% 
%if nargin < 4
%   GroundThreshold = 0.5;
%   SkyThreshold = 0.5;
%elseif nargin < 5
%   SkyThreshold = 0.5;
%end

NuRow = Default.NuRow_default;
batchRow = [1:Default.WeiBatchSize:NuRow NuRow+1];

            %==================
            maskGvec=[];
            maskSkyvec=[];

            for WeiBatchNumber = 1:floor(NuRow/Default.WeiBatchSize)              
              
              count=1;
%              for i = batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+Default.WeiBatchSize-1,NuRow)
              for i = batchRow(WeiBatchNumber):batchRow(WeiBatchNumber+1)-1
                  
                % constructing features for each batch of rows from batch featuresa
                RowskyBottom = ceil(NuRow/2);
                PatchSkyBottom = ceil(Default.VertYNuDepth*(1-Default.Horizon));
                if i <= RowskyBottom
                   PatchRowRatio = PatchSkyBottom/RowskyBottom;
                   RowTop = ceil((i-1)*PatchRowRatio+1);
                   RowBottom = ceil(i*PatchRowRatio);
                else
                   PatchRowRatio = (Default.VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
                   RowTop = ceil((i-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
                   RowBottom = ceil((i-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
                end
                ColumnLeft = 1;
                ColumnRight = Default.HoriXNuDepth;

                FeaVector = genFeaVectorNew( Default, f, FSup,...
                     [RowTop:RowBottom],[ColumnLeft:ColumnRight], 1, 0); %Notice LearnNear is 0;
                % load the GroundSkyPara for each WeiBatchNumber
                load([Default.ParaFolder '/GrndSkyTheta_Train400_WeiBatNu' num2str(WeiBatchNumber) '.mat']);
              
                ab=thetaG{count}';
                cd=thetaS{count}';
                maskGvec=[maskGvec; ab*[ones(1,305); FeaVector]];
                maskSkyvec=[maskSkyvec; cd*[ones(1,305); FeaVector]];
                count=count+1;
              end 
            end
            maskgD=maskGvec;
            maskSkyD=maskSkyvec;
            maskg=(1./(1+exp(-maskGvec)))>Default.GroundThreshold;
            maskSky=(1./(1+exp(-maskSkyvec)))>Default.SkyThreshold;
   return;


