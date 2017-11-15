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
function []=gen_Popup(sigm,k,minValue,MaxSize);

% This function generate the photopopup from the cmu software 
% for comparison purpose
% Also it generate the labeled image fro sky and ground

% Not to be used during inference.
% Generate Sky Mask using our features, and their+our data 

%%%
% For each image in filename, run the CMU photopopup.  Read in the
% labeling and extract ground and sky labels.  save both the full
% size and VertYNuDepth x HoriXNuDepth labels to
% scratch/data/MaskGSky.mat and MaskGSkyMaxSize.mat
%%%

% declaim global variable
global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize PopUpVertY PopUpHoriX taskName;

RunPopUp = true;
%load([ScratchDataFolder '/data/MaskGSky.mat']);

% prepare env setting for runing CMU's software
%origHome=matlabroot;
system('setenv MATLABROOT /afs/cs.stanford.edu/package/matlab-r14sp3/matlab/r14sp3');
system('setenv LD_LIBRARY_PATH $MATLABROOT/sys/os/glnx86:$MATLABROOT/bin/glnx86:$MATLABROOT/sys/java/jre/glnx86/jre1.5.0/lib/i386/native_threads:$MATLABROOT/sys/java/jre/glnx86/jre1.5.0/lib/i386/client:$MATLABROOT/sys/java/jre/glnx86/jre1.5.0/lib/i386');
system('setenv XAPPLRESDIR $MATLABROOT/X11/app-defaults');

% change directory to run CMU's code
cd([ScratchDataFolder '/../Popup']);

NuPics = size(filename,2);
for i = 1:NuPics
    %newfilename = strrep(filename{i},'.jpg','');
    newfilename = strrep(filename{i},'.','');
    % running the cmu software in unix
 if RunPopUp
    unix(['mkdir ' ScratchDataFolder '/popup/' filename{i}]);
    system([LocalFolder '/../third_party/PhotoPopUp/photoPopup ./classifiers_08_22_2005.mat ' '../' taskName '/ppm/' newfilename '.jpg ' ' ppm ' ScratchDataFolder '/popup/' filename{i}]);
 end   
    % start getting out the sky and ground mask
    %LableImgName = strrep(newfilename,'.','');
    imgLabel = imread([ScratchDataFolder '/popup/' filename{i} '/' newfilename '.l.jpg'],'jpg');
    maskgTemp = imgLabel(:,:,2)>imgLabel(:,:,1)& imgLabel(:,:,2)>imgLabel(:,:,3) & imgLabel(:,:,2)>100;
    maskSkyTemp = imgLabel(:,:,3)>imgLabel(:,:,1)& imgLabel(:,:,3)>imgLabel(:,:,2) & imgLabel(:,:,3)>100;
%    if MaxSize ==0
       maskg{i} = imresize(maskgTemp,[VertYNuDepth HoriXNuDepth],'nearest');
       maskSky{i} = imresize(maskSkyTemp,[VertYNuDepth HoriXNuDepth],'nearest');
%       save([ScratchDataFolder '/data/MaskGSky.mat'],'maskg','maskSky'); 
%    else
       maskgMaxSize{i} = maskgTemp;
       maskSkyMaxSize{i} = maskSkyTemp;
%       save([ScratchDataFolder '/data/MaskGSkyMaxSize.mat'],'maskgMaxSize','maskSkyMaxSize');
%    end
    if RunPopUp
       % delete the ppm file of superpixel
%       delete([ScratchDataFolder '/ppm/' newfilename '.ppm']);
       delete([ScratchDataFolder '/ppm/' newfilename '.jpg']);
    end
end
%if MaxSize ==0
   save([ScratchDataFolder '/data/MaskGSky.mat'],'maskg','maskSky'); 
%else
   save([ScratchDataFolder '/data/MaskGSkyMaxSize.mat'],'maskgMaxSize','maskSkyMaxSize');
%end

cd([LocalFolder]);
