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
function supNew = userReadImage(image_name, img_label)
% function userReadImage read the original image and the manually labeled 
% image segmentation and compute the corresponding edge image.
% input: image_name -- the name of the original image
%        image_label -- labeled image
% by Nan Hu on 10/31/2007

% load the images
orgim = imread(image_name);
labim = imread(img_label);

% adding paths
addpath(genpath('../../ec2/bin/mex'));

% get the segmented image
sup = gen_Sup_efficient_mod(orgim);

% get the indexed label image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numSeg = 4; % this is manually input, needs improve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labels = get_indexed(labim,numSeg);
labels = imresize(labels,size(sup),'nearest');

% merge superpixels with the same labels
[supNew thrd] = merge_sup_label(sup,labels);     %needs to be improved.........

% generate edge map with 1 indicate the edge point
% edgemap = gen_edgemap(sup, thrd);
% imshow(edgemap);

subplot(2,2,1), imshow(orgim);
subplot(2,2,2), imshow(labim);
subplot(2,2,3), imagesc(sup);
subplot(2,2,4), imagesc(supNew);

        
%==========================================================================
%subfunctions
%--------------------------------------------------------------------------
function labelimg = get_indexed(img,numSeg)
%tmp = rgb2hsv(img);
color = sum(img,3); %color = tmp(:,:,3);             %find the color info in the img
sizeColor = size(color);
labelimg = zeros(sizeColor);
numColor = numel(color);
color = reshape(color,numColor,1);
clear tmp;
for i=1:numSeg
    [label, freq] = mode(color(logical(color)));    %   extract the i-th most frequent color rather than zero
    labelimgtmp = zeros(sizeColor);
    if freq > numColor/100
        tmp = logical(abs(color - label)<1);      %   find the elements with the same color
        labelimgtmp(tmp)=i;                    % set an index for this color
        color(tmp)=0;                       % get rid of this color, for finding the next
    end
    %cleaning singular points
    labelimgtmp = medfilt2(labelimgtmp,[3,3]);      %median filtering to kick off singular points
    labelimg = labelimg + labelimgtmp;
end    
        
%--------------------------------------------------------------------------
function [sup threshold] = merge_sup_label(sup,labelimg)
numSeg = unique(labelimg)';      % the total indeces
numSeg = numSeg(2:end);
indold = [];
threshold = NaN;
for i = numSeg
    tmp = ~logical(labelimg - i);       % find the elements with the index 
    suplabeltmp = unique(sup(tmp))';     % get the indeces of the covered superpixels
    if numel(suplabeltmp)>1             % merge the index of those superpixels
        ind = median(suplabeltmp);
        for j = suplabeltmp
            if ~(j==ind)
                sup(~logical(sup-j))=ind;
            end
        end
    end
    if ~isempty(indold)
        threshold = min(threshold,abs(ind-indold)); %threshold is used in edge detection
    end
    indold = ind;
    clear tmp suplabeltmp;
end
threshold = threshold/max(sup(:));        
%--------------------------------------------------------------------------
function edgemap = gen_edgemap(sup,thrd)
if nargin<2
    edgemap = edge(sup,'canny');
elseif nargin==2
    edgemap = edge(sup,'canny',thrd);
end
