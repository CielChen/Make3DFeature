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
function [im, SupNeighborTable]=premergAllsuperpixel_efficient(im, Default)
%(This is program conver a non_ordered non_connected superpixel image to 
% ordered superpixel image
% input:
% im = N by M matrix depends on the image size       
% Default - Default.SmallThre = the samllest Sup size

% output:
% im_order = N by M matrix which is ordered

%%%
% For each of the superpixel indicies, finds all of the
% disconnected components with that label.  if the component is
% small (< 200 pixels) or isn't the biggest component with that
% index, call analysesupinpatch with the outline of the component
% to replace it's label with the one that is most common in the outline
%%%

%最小超像素大小
if nargin <2
   Default.SmallThre = 5; %smallest sup size
end
SupNeighborTableFlag = 1;

[yn xn] = size(im);
%超像素标号，从1开始，可能不连续，因为前面进行了下采样
NuSup = unique(im(:))';
%膨胀因子，用于把较小超像素的掩码膨胀
SE = strel('octagon',3);  
%依次处理每个超像素块
for i=NuSup

        % label connected component
        temp = zeros(size(im));
        %将所有指定超像素索引的位置标为1
        temp(im(:,:)==i)=1;
        %查找四邻域连通性，返回联通区域个数，最终矩阵，按连通区域个数对每个区域从1开始标志
        % 返回一个和temp大小相同的L矩阵，包含了标记了temp中每个连通区域的类别标签，
        %这些标签的值为1、2、num（连通区域的个数）。n的值为4或8，表示是按4连通寻找区域，还是8连通寻找，默认为8。
        [L,num] = bwlabel(temp,4);

        % find the main piece
        %返回在某一数组或数据区域中出现频率最多的数值。
        %出现频率最多的值和出现的次数
         [maxL dum]= mode(L(L~=0));
%        his = histc(L(:), 1:num);
%        [dum maxL ]= max(his);

           if dum > Default.SmallThre;
               %判断2个数组中不同元素，返回在A中有，而B中没有的值，结果向量将以升序排序返回。
               %这个判断使得只保留该标号的最大的联通区域，其他相同标号的联通区域都通过膨胀成临近的其他标号，
               %这个处理似乎不是太好？？？
              SupMerg = setdiff(1:num,maxL);
           else
              SupMerg = 1:num;
           end

           %对于每一个联通区域
           for k = SupMerg
               %设置该联通区域的掩码
               mask = L==k;
               % then assign those pixels to mostlikely 3 by 3 neighborhood
               %膨胀该联通区域
                mask_dilate = imdilate(mask,SE);
%                mask_dilate = mask | [zeros(yn,1) mask(:,1:(end-1))] ...
%                                   | [mask(:,2:(end)) zeros(yn,1)] ...
%                                   | [zeros(1,xn) ;mask(1:(end-1),:)] ...
%                                   | [mask(2:(end),:); zeros(1, xn)] ....
%                                   | [[zeros(yn-1,1) mask(2:end,1:(end-1))]; zeros(1,xn)]...
%                                   | [[mask(2:end,2:(end)) zeros(yn-1,1) ]; zeros(1,xn)]...
%                                   | [zeros(1,xn); [zeros(yn-1,1) mask(1:(end-1),1:(end-1))]]...
%                                   | [zeros(1,xn); [mask(1:(end-1),2:(end)) zeros(yn-1,1)]]...
%                                   ; 
               %将膨胀后掩码中的原始联通区域去掉
               mask_dilate(mask) = 0;
%                im(mask) = analysesupinpatch(im(mask_dilate));%hard work
               %将图像中该膨胀掩码中出现次数最多的数值赋值给联通区域，
               %这样其实就是查找联通区域指定范围内的出现最多的索引赋值给该联通区域
               im(mask) = mode(im(mask_dilate));
           end
end

% merge the small superpixel with the surrrounding one if it's neighbor is only one
 MaxSupIndex = max(NuSup(:));
 %形成索引最大值的稀疏矩阵
 SupNeighborTable = sparse(MaxSupIndex,MaxSupIndex);

 %形成超像素索引邻接矩阵
if SupNeighborTableFlag
   for i = 1:((xn-1)*yn)
     
       % registed the neoghbor in right
       %test by xiewenming 20170706
%        iright = i + yn;
%        imageI = im(i);
%        imageIright = im(iright);
%        SupNeighborTable(imageI,imageIright) = 1;
%        SupNeighborTable(imageIright,imageI) = 1;
%水平方向，正反两个方向，这样最后可以形成四邻域
       SupNeighborTable(im(i),im(i+yn)) = 1;
       SupNeighborTable(im(i+yn),im(i)) = 1;

       % registed the neoghbor in below
       if mod(i,yn) ~=0
           %test by xiewenming 20170706
%            modImageI = im(i);
%            modImageIplus1 = im(i+1);
%           SupNeighborTable(modImageI,modImageIplus1) = 1;
%           SupNeighborTable(modImageIplus1,modImageI) = 1;
%垂直方向，正反两个方向
          SupNeighborTable(im(i),im(i+1)) = 1;
          SupNeighborTable(im(i+1),im(i)) = 1;
       end
       
%         tmpsupneighbortable = SupNeighborTable(1:400, 1:400);
   end

   % find out the single neighbor ones and merge them with neighbors
   %去掉只与一个超像素有邻接关系的，因为这样的超像素属于某个超像素的怀抱中，破坏了其他超像素
   %将其填充为邻接超像素索引
   SingleTar = sum( SupNeighborTable,1);
   for i = find(SingleTar == 1)
       mask = im == i;
       im(mask) = find(SupNeighborTable(:,i) == 1);
       SupNeighborTable(:,i) = 0;
       SupNeighborTable(i,:) = 0;
   end

end
return;
