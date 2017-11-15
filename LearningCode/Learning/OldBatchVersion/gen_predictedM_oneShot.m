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
function [depthMap]=gen_predictedM_oneShot(Default, f, FeatureSuperpixel)
%learningType,logScale,SkyExclude,LearnAlg,AbsFeaType,AbsFeaDate,WeiBatchNumber,logScale,SkyExclude,LearnNear)
% this function generate the learned depth

% load all the  thi in different rows
thit = [];
for i = 1:ceil(Default.NuRow_default/Default.WeiBatchSize) % only consider two learning type 'Abs' = Depth 'Fractional' = FractionalRegDepth
    load([Default.DepthPara num2str(i) '.mat']);
    thit = [thit thi];
end  
thit = cell2mat(thit);
%end

        % prepare the thiMatrix
        NuRow = Default.NuRow_default;
        for i = 1:NuRow;
            RowskyBottom = ceil(NuRow/2);
            PatchSkyBottom = ceil(Default.VertYNuDepth*(1-Default.Horizon));
            if i <= RowskyBottom
                PatchRowRatio = PatchSkyBottom/RowskyBottom;
                RowTop(i) = ceil((i-1)*PatchRowRatio+1);
                RowBottom(i) = ceil(i*PatchRowRatio);
            else
                PatchRowRatio = (Default.VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
                RowTop(i) = ceil((i-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
                RowBottom(i) = ceil((i-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
            end
        end
        RowNumber = RowBottom'-RowTop'+1;
        thiRow = [];
        for i = 1:NuRow;
            thiRow = [ thiRow thit(:,i*ones(RowNumber(i),1))];
        end    
        %================
        FeaVectorPics = genFeaVectorNew(Default, f,FeatureSuperpixel,...
                     [1:Default.VertYNuDepth],[1:Default.HoriXNuDepth],1,0);
            depthMap = exp(reshape(sum([ones(size(FeaVectorPics,2),1) FeaVectorPics'].*...
                   repmat(thiRow',[Default.HoriXNuDepth 1]),2),Default.VertYNuDepth,[]));
return;
