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
function [Predicted]=gen_predicted(Default, f, FeatureSuperpixel, Select)
%learningType,logScale,SkyExclude,LearnAlg,AbsFeaType,AbsFeaDate,WeiBatchNumber,logScale,SkyExclude,LearnNear)
% this function generate the learned depth


FeaVectorPics = genFeaVectorNew(Default, f,FeatureSuperpixel,...
                [1:Default.VertYNuDepth],[1:Default.HoriXNuDepth],1,0);

for Option = Select

    if Option == 1
       % load all the  thiRow in different rows
       load([Default.ParaFolder 'Depth.mat']);
       Predicted.depthMap = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                            repmat(thiRow',[Default.HoriXNuDepth 1]),2),Default.VertYNuDepth,[]));
       
    elseif Option == 2
       % load all the VarRow in different rows
       load([Default.ParaFolder  'Var.mat']);
       Predicted.VarMap = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                          repmat(VarRow',[Default.HoriXNuDepth 1]),2),Default.VertYNuDepth,[]));

    else
       % load all the GSRow in different rows
       load([Default.ParaFolder 'Ground.mat']);
       load([Default.ParaFolder 'Sky.mat']);
       Predicted.Ground = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                          repmat( GroundRow',[Default.HoriXNuDepth 1]),2),Default.VertYNuDepth,[]));
       Predicted.Sky = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                          repmat( GroundRow',[Default.HoriXNuDepth 1]),2),Default.VertYNuDepth,[]));
       Predicted.Ground = (1./(1+exp(-Predicted.Ground)))>0.5;
       Predicted.Sky = (1./(1+exp(-Predicted.Sky)))>0.5;

    end  

end
return;
