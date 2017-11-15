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
%Bacth Reconstruction reconstructs all found images to wrl as per the given
%folder path
function BatchReconstruct(folderPath);


fprintf('\nAdding Paths to MATLAB');

%Add paths for the OneShotBatchRecon.m file to be run later
addpath(genpath('../../LearningCode/'));
addpath(genpath('../../third_party'));
addpath(genpath('../../bin/mex'));

%Displays the starting message and the name of the base directory
fprintf('\n\n\nStarting Batch 3D Reconstruction in the directory %s\n\n' , folderPath);

%Retrieve the directoty structure of the source folder
directoryStructure = dir(folderPath);

%Iterate over each directory found in the source folder. Start is at three
%to avoid the folder entries of "." and ".."
%************************************************** I Loop
for i=3:length(directoryStructure)
   
    fprintf('\n\n%s\n\n',directoryStructure(i).name);
    
    levelTwoPath = strcat(folderPath , '/' ,  directoryStructure(i).name);
   
    subLevelDirStruct = dir(levelTwoPath);
    
    %*********** J Loop
   for j=3:length(subLevelDirStruct)
       
       levelThreePath = strcat(levelTwoPath , '/', subLevelDirStruct(j).name);
       
       fprintf('%s\n',subLevelDirStruct(j).name);
       
       imageLevelStructure = dir(levelThreePath);
      
       %compute full path of the image to be reconstructed
       imageFilename = strcat(levelThreePath , '/' , imageLevelStructure(3).name);
                    
       %compute full output path
       outputFolder = strcat(levelThreePath , '/');
     
       fprintf('%s -- %s\n' , imageFilename , outputFolder);
       
       %****** IF
       if( length(imageLevelStructure) == 3 )
         
         OneShotBatchRecon(imageFilename , outputFolder);
           
       end
       %***** ENDIF
   end
   %************ J Loop
     
end
%*************************************************** I Loop
