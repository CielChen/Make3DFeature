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
function []=Depth_error(DepthDirectory, SkyExclude, detail,baseline)

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

% prepare report
ReportName = [ScratchDataFolder '/result/DepthErrorTable.txt' ];
check = dir(ReportName)
DateStamp = date;
if isempty(check)
   disp('is empty')
   fid = fopen(ReportName, 'w+');
   fprintf(fid, '%s %s %s %s %s %s %s\n', DateStamp,'MeanAbsError','RMSError','MeanAbsLogError','RMSLogError','MeanAbsFraError','RMSFraError');
   fclose(fid);
end
DepthDirectory
check = dir([DepthDirectory]);
if isempty(check)
   disp('empty')
end
%return;

NuPics = size(filename,2);
load([ScratchDataFolder '/data/MaskGSky.mat']);
l =1;
PicsInd = [19:60 109:134]  % for Ashutosh preivious work
%for i= 1:NuPics
for i= PicsInd
    % in case can't find the laser depth
    depthfile = strrep(filename{i},'img','depth_sph_corr');
    check = dir([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
    if isempty(check)
       continue;
    end
    load([ScratchDataFolder '/Gridlaserdata/' depthfile '.mat']);
    laserdepth = Position3DGrid(:,:,4);
    clear Position3DGrid;

    % load learned or predicted depth
    depthfile = strrep(filename{i},'img','depth_learned');
    if baseline ==1
       disp('load baseline')
       check = dir([DepthDirectory '_baseline/' depthfile '.mat'])
       if isempty(check)
          continue;
       end
       load([DepthDirectory '_baseline/' depthfile '.mat']);
       depthMap = depthMap_base;
    elseif baseline == 2
       disp('load baseline2')
       check = dir([DepthDirectory '_baseline2/' depthfile '.mat'])
       if isempty(check)
          continue;
       end
       load([DepthDirectory '_baseline2/' depthfile '.mat']);
       depthMap = depthMap_base2;
    else
       check = dir([DepthDirectory '/' depthfile '.mat'])
       if isempty(check)
          continue;
       end
       load([DepthDirectory '/' depthfile '.mat']);
    end

    i
    filename{i}    
    % calculate average abs error RMS error on linear and log scale, and fractional error
    if SkyExclude == 1
       disp('Sky Exlude')
 
       % remove big error so that it won't effect linear and fractional term
       AbsLogError=abs( log10(depthMap(:))-log10(laserdepth(:)) );
       %AbsLogError = AbsLogError(mask); % get rid of the sky
       % do auto scaling =============like for CMU
         if median(depthMap(:)) >20
            continue;
         end
         scale = median(AbsLogError(:));
         depthMap = depthMap*exp(scale); 
%        laserdepth = laserdepth*exp(-scale);
         
       % ========================================
       maskBigError = (depthMap >70); % 80+30
%       maskBigError = (depthMap >Inf); % 80+30

       mask = ~(maskSky{i}|maskBigError);
       sum(mask(:))
       AbsLogError=abs( log10(depthMap(:))-log10(laserdepth(:)) );
       AbsLogError = AbsLogError(mask); % get rid of the sky
       AbsError=abs( (depthMap(:))-(laserdepth(:)) );
       AbsError = AbsError(mask);
       AbsFraError = abs( (depthMap(:)-laserdepth(:))./laserdepth(:));
       AbsFraError = AbsFraError(mask);
    else
       AbsLogError=abs( log10(depthMap(:))-log10(laserdepth(:)) );
       AbsError=abs( (depthMap(:))-(laserdepth(:)) );
       AbsFraError = abs( (depthMap(:)-laserdepth(:))./laserdepth(:));
    end
        

    WholeAbsLogError{l} = AbsLogError;
    WholeAbsError{l} = AbsError;
    WholeAbsFraError{l} = AbsFraError;
    l = l +1;

end

MeanAbsError = mean(cell2mat(WholeAbsError'));
RMSError = sqrt(mean(cell2mat(WholeAbsError').^2));
MeanAbsLogError = mean(cell2mat(WholeAbsLogError'));
RMSLogError = sqrt(mean(cell2mat(WholeAbsLogError').^2));
MeanAbsFraError = mean(cell2mat(WholeAbsFraError'));
RMSFraError = sqrt(mean(cell2mat(WholeAbsFraError').^2));
if detail
   disp('report detail');
   save([ScratchDataFolder '/result/DepthErrorReport' DateStamp '.mat'],...
   'WholeAbsError','WholeAbsLogError','WholeAbsFraError','MeanAbsError','RMSError','MeanAbsLogError','RMSLogError','MeanAbsFraError','RMSFraError');
end

fid = fopen(ReportName, 'a+');
fprintf(fid, '%s %g      %g  %g        %g    %g        %g\n %s %d\n', DateStamp,MeanAbsError,RMSError,MeanAbsLogError,RMSLogError,MeanAbsFraError,RMSFraError, DepthDirectory, baseline);
fclose(fid);


