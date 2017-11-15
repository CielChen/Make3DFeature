# *  This code was used in the following articles:
# *  [1] Learning 3-D Scene Structure from a Single Still Image, 
# *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
# *      In ICCV workshop on 3D Representation for Recognition (3dRR-07), 2007.
# *      (best paper)
# *  [2] 3-D Reconstruction from Sparse Views using Monocular Vision, 
# *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
# *      In ICCV workshop on Virtual Representations and Modeling 
# *      of Large-scale environments (VRML), 2007. 
# *  [3] 3-D Depth Reconstruction from a Single Still Image, 
# *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
# *      International Journal of Computer Vision (IJCV), Aug 2007. 
# *  [6] Learning Depth from Single Monocular Images, 
# *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
# *      In Neural Information Processing Systems (NIPS) 18, 2005.
# *
# *  These articles are available at:
# *  http://make3d.stanford.edu/publications
# * 
# *  We request that you cite the papers [1], [3] and [6] in any of
# *  your reports that uses this code. 
# *  Further, if you use the code in image3dstiching/ (multiple image version),
# *  then please cite [2].
# *  
# *  If you use the code in third_party/, then PLEASE CITE and follow the
# *  LICENSE OF THE CORRESPONDING THIRD PARTY CODE.
# *
# *  Finally, this code is for non-commercial use only.  For further 
# *  information and to obtain a copy of the license, see 
# *
# *  http://make3d.stanford.edu/publications/code
# *
# *  Also, the software distributed under the License is distributed on an 
# * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# *  express or implied.   See the License for the specific language governing 
# *  permissions and limitations under the License.
# *
# */
#!/bin/bash

#required
IMG_PATH=$1
OUTPUT_FOLDER=$2

#optional
TASK_NAME=$3
SCRATCH_FOLDER=$4
PARAMS_FOLDER=$5

export matlabHome="/afs/cs.stanford.edu/package/matlab-r2006b/matlab/r2006b"

#setenv LD_LIBRARY_PATH
${matlabHome}/sys/os/glnxa64:${matlabHome}/bin/glnxa64:${matlabHome}/sys/java/jre/glnxa64/jre1.5.0/lib/amd64/native_threads:${matlabHome}/sys/java/jre/glnxa64/jre1.5.0/lib/amd64/client:${matlabHome}/sys/java/jre/glnxa64/jre1.5.0/lib/amd64/server:${matlabHome}/sys/java/jre/glnxa64/jre1.5.0/lib/amd64:${matlabHome}/sys/opengl/lib/sys/opengl/lib/glnxa64:${LD_LIBRARY_PATH}


#export
LD_LIBRARY_PATH=${matlabHome}/sys/os/glnxa64:${matlabHome}/bin/glnxa64:${LD_LIBRARY_PATH}

export
LD_LIBRARY_PATH=${matlabHome}/sys/os/glnx86:${matlabHome}/bin/glnx86:${LD_LIBRARY_PATH}
export
LD_LIBRARY_PATH="/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/mex":${LD_LIBRARY_PATH}
export
LD_LIBRARY_PATH="/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/oneShot":${LD_LIBRARY_PATH}
export PATH="/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/mex":${PATH}
export PATH="/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/oneShot":${PATH}
export XAPPLRESDIR=${matlabHome}/X11/app-defaults


echo ${LD_LIBRARY_PATH}
echo ${PATH}

#export PATH={$PATH}:/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/mex
#export
LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:/afs/cs.stanford.edu/u/sidbatra/Recon3d/bin/mex

#setenv LD_LIBRARY_PATH ${homeFolder}/Inference:${LD_LIBRARY_PATH}
#setenv solverHome /afs/cs/group/reconstruction3d/Data
#setenv LD_LIBRARY_PATH
${solverHome}/yalmip:${solverHome}/SeDuMi_1_1R3/SeDuMi_1_1:${LD_LIBRARY_PATH}

export LD_PRELOAD=/usr/lib/libstdc++.so.6
export MATLAB_SHELL=/bin/sh

./OneShot3dEfficient $IMG_PATH $OUTPUT_FOLDER
