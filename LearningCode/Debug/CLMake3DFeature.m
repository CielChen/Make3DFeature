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
function CLMake3DFeature(ImgPath, OutPutFolder,...
     taskName,...% taskname will append to the imagename and form the outputname
     ScratchFolder,... % ScratchFolder
     ParaFolder,... % All Parameter Folder
     Flag...  % All Flags 1) intermediate storage flag
     );

if ~isdeployed
  addpath(genpath('../../LearningCode'));
 addpath(genpath('../../third_party'));
 addpath(genpath('../../bin/mex'));
end

% This function is the speed up version of the OneShot3d
% Improvement Log:
% 1) speedup segment Mex-file
% 2) speedup SparseSample3d Mex-file
% 3) eliminate reading image and filterbank calculation multiple times

% Input:
% ImgPath -- the path include the file name of the image
% OutPutFolder -- the path of the output folder
% ScratchFolder -- intermediante data storage place (used for learning and debug)

% Parameter and Data Setting =========================
  startTime = tic;
  fprintf('Begin Make3D feature...                ');
  fprintf('Starting with new optimization...                        ');

  if nargin < 2
     disp('Eror: At least need two input argument needed');
     return;
  elseif nargin < 3
     taskName = '';
     Flag = [];
     %mod by xiewenming 20170629
%      ScratchFolder = ['/afs/cs/group/reconstruction3d/scratch/IMStorage' ];
%      ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/';
     ScratchFolder = ['../../make3d/scratch' ];
     ParaFolder = '../../make3d/params/';
  elseif nargin < 4
     Flag = [];
     %mod by xiewenming 20170629
%      ScratchFolder = ['/afs/cs/group/reconstruction3d/scratch/IMStorage' ];
%      ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/';
     ScratchFolder = ['../../make3d/scratch' ];
     ParaFolder = '../../make3d/params/';
  elseif nargin < 5
     Flag = [];
     %mod by xiewenming 20170629
%      ParaFolder = '/afs/cs/group/reconstruction3d/scratch/Para/';
     ParaFolder = '../../make3d/params/';
  elseif nargin < 6
     Flag = [];
  end

%	yalmiptest

% parameter setting
  filename{1} = ImgPath( ( max( strfind( ImgPath, '/'))+1) :end);

  % Function that setup the Default
  %设置默认参数
  Default = SetupDefault_New(...
            [ strrep(filename{1}, '.jpg', '') '_' taskName],...
            ParaFolder,...
            OutPutFolder,...
            ScratchFolder,...
            Flag);
  disp([ num2str( toc(startTime) ) ' seconds.']);
  
  %读取标注文件：图片名，光源索引，左上角坐标（x），左上角坐标（y），宽，高
  [ImgName,ImgIdx,ImgLeft,ImgTop,ImgWidth,ImgHeight] = textread(...
      '../LightImages/LightLable.txt',...
      '%s%n%n%n%n%n');
    
%     [ImgName,ImgIdx,ImgLeft,ImgTop,ImgWidth,ImgHeight] = textread(...
%       '../LightImages/LightLable.txt',...
%       '%s%n%n%n%n%n','headerlines',6);
%   
  %依次获取文件夹下所有图片提取特征，并保存到相同文件名的文本文件中
  Files = dir(fullfile('../LightImages/', '*.jpg'));
  LengthFiles = length(Files);
  for iFile = 1:LengthFiles
      Default.filename{1} = [ strrep(Files(iFile).name, '.jpg', '') '_' taskName];
      
      % Image loading
      % 读取图片
      fprintf('Loading the image %s...', Files(iFile).name);
      % img = imread(ImgPath);
      img = imread(strcat('../LightImages/', Files(iFile).name));
      %记录原始图像的大小
      [clRawImgRow,clRawImgCol,clRawImgPixel]= size(img);
      %查找该图片在光源标注列表中的索引
      clLightIdx = find(strcmp(ImgName, Files(iFile).name));
      %光源标注位置矩阵，依次交替xy值
      clLightSize = length(clLightIdx);
      clLightPosMat = zeros(clLightSize * 2, 5);
      for iLight = 1:clLightSize
          tmpLeft = ImgLeft(clLightIdx(iLight));
          tmpTop = ImgTop(clLightIdx(iLight));
          tmpRight = tmpLeft + ImgWidth(clLightIdx(iLight));
          tmpDown = tmpTop + ImgHeight(clLightIdx(iLight));
          iTwoLight = 2 * iLight;
          clLightPosMat(iTwoLight-1,:) = [tmpLeft tmpRight tmpRight tmpLeft tmpLeft];
          clLightPosMat(iTwoLight,:) = [tmpTop tmpTop tmpDown tmpDown tmpTop];
      end
      
      %  imgCameraParameters = exifread(ImgPath);
      %	if false %Default.Flag.DisplayFlag && (any( strcmp(fieldnames(imgCameraParameters),'FocalLength') ) || ...
      %						  any( strcmp(fieldnames(imgCameraParameters),'FNumber') )  	|| ...
      %						  any( strcmp(fieldnames(imgCameraParameters),'FocalPlaneXResolution') )	|| ...
      %						  any( strcmp(fieldnames(imgCameraParameters),'FocalPlaneYResolution') ) )
      % FocalPlaneResolutionUnit
      %		disp('This image has known  f  and/or   f/sx ');
      %	end
      disp([ num2str( toc(startTime) ) ' seconds.']);
      
      % ***************************************************      
      % Features ===========================================      
      % 1) Basic Superpixel generation and Sup clean
      % 产生超像素
      fprintf('Creating Superpixels...           ');
      [MedSup, Sup, Default, SupNeighborTable] = gen_Sup_efficient(Default, img);
      disp([ num2str( toc(startTime) ) ' seconds.']);
      
      % 2) Texture Features and inner multiple Sups generation
      %   load /afs/cs/group/reconstruction3d/scratch/Train400/data/MaskGSky.mat;
      %   load /afs/cs/group/reconstruction3d/scratch/Train400/data/LowResImgIndexSuperpixelSep.mat;
      %   load /afs/cs/group/reconstruction3d/scratch/Train400/data/MedSeg/MediResImgIndexSuperpixelSep1.mat
      %   maskg = maskg{1};
      %   [TextureFeature TextSup]=GenTextureFeature_InnerMulSup(Default, img, Sup{2}, LowResImgIndexSuperpixelSep{1},...
      %                            imresize((MediResImgIndexSuperpixelSep),[Default.TrainVerYSize Default.TrainHoriXSize],'nearest'), 1, maskg);
      % comment compare with old value different only in 1:34 features since
      % superpixel changes
      fprintf('Creating Features and multiple segmentations... ');
      [TextureFeature TextSup]=GenTextureFeature_InnerMulSup(Default, img, Sup{2}, Sup{1},...
          imresize((MedSup),[Default.TrainVerYSize Default.TrainHoriXSize],'nearest'), 1);%, maskg);
      disp([ num2str( toc(startTime) ) ' seconds.']);
      
      %组织成所需的特征值格式
      %不同等级的超像素的索引不一致，这里先给最高等级的超像素及其特征
      %输出：[超像素的索引，是否光照区域，中心点位置（行列），34特征（中心，上下左右），邻接表]
      %[超像素索引表][超像素邻接关系稀疏矩阵]
      %超像素标号，增序
      clSupImg = Sup{1};
      %记录超像素图像的大小
      [clSupImgRow,clSupImgCol,clSupImgPixel] = size(clSupImg);
      clScaleSup2RawRow = clRawImgRow / clSupImgRow;
      clScaleSup2RawCol = clRawImgCol / clSupImgCol;
      %形成索引
      clSupIndex = unique(clSupImg(:));
      clIndexMatrix = zeros(size(clSupIndex, 1), 344);
      %第一列：超像素的索引
      clIndexMatrix(:,1) = clSupIndex;
      %第二列：是否是光源
%       clIndexMatrix(:,2) = 0;
      %第三列、第四列：超像素的中心点位置，行号和列号
      %转化为原始图像的位置？？
      idxSup = 1;
      clSupIndexRow = clSupIndex';
      for isup = clSupIndexRow
          mask = clSupImg==isup;
          [x,y] = find(mask);
          finalX = int64(prctile(x, 50));
          finalY = int64(prctile(y, 50));
          %转化为原始图像的位置，再进行存储
          finalRawX = finalX * clScaleSup2RawRow;
          finalRawY = finalY * clScaleSup2RawCol;
          clIndexMatrix(idxSup, 3) = floor(finalRawX);
          clIndexMatrix(idxSup, 4) = floor(finalRawY);
          
          %判断超像素每行的中心是否处于光源范围内
          UniX = unique(x);
          UniXRow = UniX';
          bEndLightLoop = 0;
          for iUniX = UniXRow
              PixIdxX = find(x == iUniX);
              PixelY = y(PixIdxX');
              
              %判断中心位置
              PixelYMedian = int64(median(PixelY));
              %转化为原始图像的位置，再进行存储
              finalMedianX = int64(iUniX) * clScaleSup2RawRow;
              finalMedianY = PixelYMedian * clScaleSup2RawCol;
              %判断超像素每行的中心是否处于光源范围内
              for iLight = 1:clLightSize
                  in = inpolygon(double(finalMedianX), double(finalMedianY),...
                      clLightPosMat(iLight,:), clLightPosMat(iLight+1,:));
                  if (in == 1)
                      clIndexMatrix(idxSup, 2) = 1;
                      bEndLightLoop = 1;
                      break;
                  end
              end              
              if bEndLightLoop == 1
                  break;
              end
              
              %判断最左边点
              PixelYLeft = PixelY(1);
              %转化为原始图像的位置，再进行存储
              finalLeftX = int64(iUniX) * clScaleSup2RawRow;
              finalLeftY = int64(PixelYLeft) * clScaleSup2RawCol;
              %判断超像素每行的中心是否处于光源范围内
              for iLight = 1:clLightSize
                  in = inpolygon(double(finalLeftX), double(finalLeftY),...
                      clLightPosMat(iLight,:), clLightPosMat(iLight+1,:));
                  if (in == 1)
                      clIndexMatrix(idxSup, 2) = 1;
                      bEndLightLoop = 1;
                      break;
                  end
              end              
              if bEndLightLoop == 1
                  break;
              end
              
              %判断最右边点
              lenPixelY = length(PixelY);
              PixelYRight = PixelY(lenPixelY);
              %转化为原始图像的位置，再进行存储
              finalRightX = int64(iUniX) * clScaleSup2RawRow;
              finalRightY = int64(PixelYRight) * clScaleSup2RawCol;
              %判断超像素每行的中心是否处于光源范围内
              for iLight = 1:clLightSize
                  in = inpolygon(double(finalRightX), double(finalRightY),...
                      clLightPosMat(iLight,:), clLightPosMat(iLight+1,:));
                  if (in == 1)
                      clIndexMatrix(idxSup, 2) = 1;
                      bEndLightLoop = 1;
                      break;
                  end
              end              
              if bEndLightLoop == 1
                  break;
              end              
          end
          
          %判断2
%           %判断超像素每行的中心是否处于光源范围内
%           UniX = unique(x);
%           UniXRow = UniX';
%           bEndLightLoop = 0;
%           for iUniX = UniXRow
%               PixIdxX = find(x == iUniX);
%               PixelY = y(PixIdxX');
%               PixelYMedian = int64(median(PixelY));
%               %转化为原始图像的位置，再进行存储
%               finalMedianX = int64(iUniX) * clScaleSup2RawRow;
%               finalMedianY = PixelYMedian * clScaleSup2RawCol;              
% 
%               %判断超像素每行的中心是否处于光源范围内
%               for iLight = 1:clLightSize
%                   in = inpolygon(double(finalMedianX), double(finalMedianY),...
%                       clLightPosMat(iLight,:), clLightPosMat(iLight+1,:));
%                   if (in == 1)
%                       clIndexMatrix(idxSup, 2) = 1;
%                       bEndLightLoop = 1;
%                       break;
%                   end
%               end
%               
%               if bEndLightLoop == 1
%                   break;
%               end
%           end
    
          %判断3
%           %判断超像素中心是否处于光源范围内
%           for iLight = 1:clLightSize
%               in = inpolygon(double(finalRawX), double(finalRawY),...
%                   clLightPosMat(iLight,:), clLightPosMat(iLight+1,:));
%               if (in == 1)
%                   clIndexMatrix(idxSup, 2) = 1;
%                   break;
%               end
%           end
          
          %68维特征，依次是中心，左上右下
          %中心68维特征
          feaLine = finalX * finalY;
          %       if (feaLine <= 0)
          %           hahaerror = 0;
          %       else
          %           heheright = 1;
          %       end
          clIndexMatrix(idxSup, 5:72) = TextureFeature.Abs(feaLine, 2:69);
          %查找邻接索引，这里先查找所有以当前索引开始的所有索引，选择其中最先的四个按照左上右下的顺序赋值
          clNeighborMat = SupNeighborTable(isup, :);
          clNeighborIdx = find(clNeighborMat==1);
          clNeighborTrueIdx = find(clNeighborIdx ~= isup);
          clNeighborSize = length(clNeighborTrueIdx);
          if (clNeighborSize > 0)
              %左
              maskLeft = clSupImg==clNeighborIdx(clNeighborTrueIdx(1));
              [xLeft yLeft] = find(maskLeft);
              finalXLeft = int64(prctile(xLeft, 50));
              finalYLeft = int64(prctile(yLeft, 50));
              feaLineLeft = finalXLeft * finalYLeft;
              clIndexMatrix(idxSup, 73:140) = TextureFeature.Abs(feaLineLeft, 2:69);
          end
          if (clNeighborSize > 1)
              %上
              maskUp = clSupImg==clNeighborIdx(clNeighborTrueIdx(2));
              [xUp yUp] = find(maskUp);
              finalXUp = int64(prctile(xUp, 50));
              finalYUp = int64(prctile(yUp, 50));
              feaLineUp = finalXUp * finalYUp;
              clIndexMatrix(idxSup, 141:208) = TextureFeature.Abs(feaLineUp, 2:69);
          end
          if (clNeighborSize > 2)
              %右
              maskRight = clSupImg==clNeighborIdx(clNeighborTrueIdx(3));
              [xRight yRight] = find(maskRight);
              finalXRight = int64(prctile(xRight, 50));
              finalYRight = int64(prctile(yRight, 50));
              feaLineRight = finalXRight * finalYRight;
              clIndexMatrix(idxSup, 209:276) = TextureFeature.Abs(feaLineRight, 2:69);
          end
          if (clNeighborSize > 3)
              %下
              maskDown = clSupImg==clNeighborIdx(clNeighborTrueIdx(4));
              [xDown yDown] = find(maskDown);
              finalXDown = int64(prctile(xDown, 50));
              finalYDown = int64(prctile(yDown, 50));
              feaLineDown = finalXDown * finalYDown;
              clIndexMatrix(idxSup, 277:344) = TextureFeature.Abs(feaLineDown, 2:69);
          end
          
          idxSup = idxSup + 1;
      end
      
      saveFile = Files(iFile).name;
      [pathstr, name, ext] = fileparts(saveFile);
%       newfilename = strcat(name,'.txt');
      newfilePathname = fullfile('../OutputTest/', strcat(name,'.txt'));
      save(newfilePathname, 'clIndexMatrix', '-ascii');
      
      fprintf('Finish the image %s...', Files(iFile).name);
  end
  
return;
