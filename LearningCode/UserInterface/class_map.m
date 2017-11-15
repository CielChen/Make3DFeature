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
clear all; close all; clc;
% This script inputs png image file from the user app and creates a 
% "class" map of the image 
[img] = imread('image_name','png','BackgroundColor',[1, 1, 1]);
[H W depth] = size(img);
bitdepth = 255;
list = [];
threshold = 20; range = 3;

%Convert to gray scale
I=rgb2gray(img);

%Compute the histogram of the flattened image
[amp, bins] = hist(I(:), -4:255);

%Get ride of the white values + zero pad the begining
amp((length(bins)-4):length(bins)) = 0;

%Gate histogram signal 
for i=1:length(amp)
    if (amp(i) < threshold)
        amp(i) = 0;
    end
end

%Find maxs until all amplitude values are zero
[maxx index] = max(amp);
while( maxx > 5)
    amp((index-range):(index+range)) = 0;
    list = [bins(index) list];
    [maxx index] = max(amp);
end
list

%Now go through entire image and label the class for the image: classmap
classmap = zeros(H,W);
for i=1:H
    for j=1:W
        for k=1:length(list)
            if  ((list(k) - range) <I(i, j)) && (I(i, j) < (list(k)+ range))
                classmap(i, j) = k;
                break;
            end
        end
    end
end

classmap

        
        
        
        
        
        
        
