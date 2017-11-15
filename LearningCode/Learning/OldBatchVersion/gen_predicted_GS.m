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
function [maskGvec, maskSkyvec]=gen_predicted_GS(TrainSet,HistFeaType,HistFeaDate,AbsFeaType,AbsFeaDate)

global GeneralDataFolder ScratchDataFolder LocalFolder ClusterExecutionDirectory...
    ImgFolder VertYNuPatch VertYNuDepth HoriXNuPatch HoriXNuDepth a_default b_default Ox_default Oy_default...
    Horizon_default filename batchSize NuRow_default SegVertYSize SegHoriXSize WeiBatchSize PopUpVertY PopUpHoriX taskName;

statusFilename = [ClusterExecutionDirectory '/matlabExecutionStatus_depth.txt'];
NuPics = size(filename,2);
NuBatch = ceil(NuPics/batchSize);
NuRow = NuRow_default;
%Horizon = Horizon_default;
%skyBottom = floor(NuRow/2);
batchRow = 1:WeiBatchSize:NuRow;
%GassuianRegularization = true;
%RegularWei = 0.01;

    load([ScratchDataFolder '/data/FeatureSuperpixel.mat']); % load the feature relate to position and shape of superpixel
    %load([ScratchDataFolder '/data/MaskGSky.mat']); % maskg is the estimated ground maskSky is the estimated sky
    % load estimated sky
    for j = 1:NuBatch
        tic
        load([ScratchDataFolder '/data/feature_Abs_' AbsFeaType int2str(j) '_' AbsFeaDate '.mat']); % 'f'
        %toc
        %for k = trainIndex{j}
        for k = 1:size(f,2)%batchSize

            %==================
            % load picsinfo just for the horizontal value
        PicsinfoName = strrep(filename{(j-1)*batchSize+k},'img','picsinfo');
        temp = dir([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
        if size(temp,1) == 0
            a = a_default;
            b = b_default;
            Ox = Ox_default;
            Oy = Oy_default;
            Horizon = Horizon_default;
        else
            load([GeneralDataFolder '/PicsInfo/' PicsinfoName '.mat']);
        end

            %RowTop=1;
            %RowBottom=VertYNuDepth;
            maskGvec=[];
            maskSkyvec=[];

            for WeiBatchNumber = 1:floor(NuRow/WeiBatchSize)
              count=1;
              for i = batchRow(WeiBatchNumber):min(batchRow(WeiBatchNumber)+WeiBatchSize-1,NuRow)
            %i=RowNumber;
            % constructing features for each batch of rows from batch featuresa
            %l
                RowskyBottom = ceil(NuRow/2);
                PatchSkyBottom = ceil(VertYNuDepth*(1-Horizon));
                if i <= RowskyBottom
                   PatchRowRatio = PatchSkyBottom/RowskyBottom;
                   RowTop = ceil((i-1)*PatchRowRatio+1);
                   RowBottom = ceil(i*PatchRowRatio);
                else
                   PatchRowRatio = (VertYNuDepth-PatchSkyBottom)/(NuRow-RowskyBottom);
                   RowTop = ceil((i-RowskyBottom-1)*PatchRowRatio+1)+PatchSkyBottom;
                   RowBottom = ceil((i-RowskyBottom)*PatchRowRatio)+PatchSkyBottom;
                end
                ColumnLeft = 1;
                ColumnRight = HoriXNuDepth;

                FeaVector = genFeaVector(f{k},FeatureSuperpixel{(j-1)*batchSize+k},...
                     [RowTop:RowBottom],[ColumnLeft:ColumnRight],(j-1)*batchSize+k,0); %Notice LearnNear is 0;
                load([ScratchDataFolder '/../learned_parameter/GrndSkyTheta_' TrainSet '_WeiBatNu' ...
                     num2str(WeiBatchNumber) '_' AbsFeaType '_AbsFeaDate' AbsFeaDate  '_LearnDate.mat']);%TestDisp.mat']); 
                %FeaWei = [];
                % DepthVector = [];
                fid = fopen(statusFilename, 'w+');
                fprintf(fid, 'Currently on row number %i\n', i);
                fclose(fid);        %file opening and closing has to be inside the loop, otherwise the file will not appear over afs
                disp(['Going to Run Step 9, WeiBatchNumber = ' num2str(WeiBatchNumber) ' i=' num2str(i) ' j=' num2str(j) ' k=' num2str(k)]);
                %thetaG
                %thetaS
                %pause
                %size(FeaVector)
%                if (WeiBatchNumber == 4 && count == 5)
%                  size(thetaG{count-1})
%                  size(thetaS{count-1})
%                  %pause
%                  ab=thetaG{count-1}';
%                  cd=thetaS{count-1}';
%                else
                  %size(thetaG{count})%
                  %size(thetaS{count})
                  %pause
                  ab=thetaG{count}';
                  cd=thetaS{count}';
%                end
                maskGvec=[maskGvec; ab*[ones(1,305); FeaVector]];
                maskSkyvec=[maskSkyvec; cd*[ones(1,305); FeaVector]];
                count=count+1;
              end 
            end
            picNumber=(j-1)*batchSize+k;
            maskgD{picNumber}=maskGvec;
            maskSkyD{picNumber}=maskSkyvec;
            maskg{picNumber}=(1./(1+exp(-maskGvec)))>0.5;
            maskSky{picNumber}=(1./(1+exp(-maskSkyvec)))>0.5;
            save([ScratchDataFolder '/data/MaskGSky.mat'],'maskg','maskSky','maskgD','maskSkyD');
            disp(['done ... uff for ' num2str(picNumber)]);
            %pause;
        end
        clear f newFea;% Position3DGrid;
        toc
    end
   


