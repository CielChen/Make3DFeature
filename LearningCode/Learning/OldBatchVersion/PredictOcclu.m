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
function []=PredictOcclusion()

% This function load Psi to predict the occlusion bounday of image
% Then show the result in Grid Boundary
% And the Positive Accuracy and Negitive Accuracy within image

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
       ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
       Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

NuPics = size(filename,2);
NuRow = NuRow_default;

% load all the Psi 
for l = 1:2 % load vertical and horizontal 
    for i = 1: NuRow
        if l == 1
           load([ScratchDataFolder '/data/AlignLearn/AlignLearnHori_' num2str(i) '.mat']);
        else
           load([ScratchDataFolder '/data/AlignLearn/AlignLearnVert_' num2str(i) '.mat']);
        end
        PsiAll{i,l} = Psi;
    end
end

for k = 1: NuPics
    LabelCand = [];
    % load the nList and the feature
    k
    load([ScratchDataFolder '/data/SupFea/FeaNList' num2str(k) '.mat']); % load nList (y3 x4 )and FeaNList	
    % evaluate the Label for the whole image
    for l = 1:2
        for i = 1: NuRow
            LabelCand(:,i+(l-1)*NuRow) = glmval( PsiAll{i,l}, FeaNList, 'logit');
        end
    end
    % pick out the correct label
    List = [ceil(nList(:,3)*VertYNuDepth) abs(nList(:,5:6)*[1 0]') <= abs(nList(:,5:6)*[0 1]')];
    List = NuRow*List(:,2)+List(:,1);
%nList
    Ind = sub2ind(size(LabelCand), (1:size(List,1))', List);
    Label = LabelCand(Ind);
    save([ScratchDataFolder '/data/occluLabel/Label' num2str(k) '.mat'],'Label','nList');
end
