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
function [t]=test_Ash_L1_Minimizer_min(filename)

global A b S inq

RealProblemFlag = 1;

if ~RealProblemFlag
   A = rand(800,500);
   b = rand(800,1);
   save temp.mat A b
   %load temp.mat
else
   %filename = 'img-combined1-p-220t0_';
   if nargin <1
   filename = 'Catsh1stOpt_GatePackard000_';
   end
%    ScratchFolder = '/afs/cs/group/reconstruction3d/scratch/IMStorage/';
%    OutPutFolder = '/afs/cs.stanford.edu/group/reconstruction3d/scratch/3DmodelMultipleImage/';
   ScratchFolder = 'C:\Documents and Settings\Ash\My Documents\Reconstruction3d\DataTemp\';
   OutPutFolder = 'C:\Documents and Settings\Ash\My Documents\Reconstruction3d\DataTemp\';
   load([ScratchFolder filename '.mat']);
   A = [ NewPosiM.*repmat(WeightsSelfTerm,1,size(NewPosiM,2));...
         NewCoPM.*repmat(NewCoPEstDepth,1,size(NewCoPM,2))*Center;...
         NewHoriStickM.*repmat( NewEstDepHoriStick.*NewWeightHoriNeighborStitch,1,size( NewHoriStickM,2));...
         NewVertStickM.*repmat( NewEstDepVertStick.*NewWeightVertNeighborStitch,1,size( NewVertStickM,2))];
A1=  NewPosiM.*repmat(WeightsSelfTerm,1,size(NewPosiM,2));
A2=  NewCoPM.*repmat(NewCoPEstDepth,1,size(NewCoPM,2))*Center;
A3=  NewHoriStickM.*repmat( NewEstDepHoriStick.*NewWeightHoriNeighborStitch,1,size( NewHoriStickM,2));
A4=  NewVertStickM.*repmat( NewEstDepVertStick.*NewWeightVertNeighborStitch,1,size( NewVertStickM,2));
   b = [ ones(size(NewPosiM,1),1);...
         NewCoPMBound.*NewCoPEstDepth*Center;...
         NewHoriStickMBound.*NewEstDepHoriStick.*NewWeightHoriNeighborStitch;...
         NewVertStickMBound.*NewEstDepVertStick.*NewWeightVertNeighborStitch];
   % inequalities
%   NuSubSupSize = size(A,2) / 3;
   temp = zeros(1, NuSubSupSize*3);
   temp(3*(1:NuSubSupSize)-1) = YPointer(Sup2Para(SubSup));
   temp = sparse(1:length(temp), 1:length(temp), temp);
   temp( sum(temp,2) ==0,:) = [];
   S = [temp;...
        NewRayAllM;...
        -NewRayAllM];
   q = [ sparse(size(temp,1), 1);...
         - 1/ClosestDist*ones(size(NewRayAllM,1),1);...
         1/FarestDist*ones(size(NewRayAllM,1),1)];
   
end
Para.A1endPt = size(A1,1);
Para.A2endPt = Para.A1endPt+size(A2,1);
Para.A3endPt = Para.A2endPt+size(A3,1);
Para.A4endPt = Para.A3endPt+size(A4,1);
Para.ClosestDist = ClosestDist;
Para.FarestDist = FarestDist;
clear A1 A2 A3 A4;
% test on ashu's code

%S = S(floor(end/2):floor(3*end/5),:);
%q = q(floor(end/2):floor(3*end/5)); 
%disp('Reducing inequalities by 1/3');

%ashIteratorTime = tic;
%[x_ashIterator, fail, info] = Ash_L1_Minimizer_min(A1, A2, A3, A4, b, 1e-12, 1);
% log_barrier (Para, A, b, S, q, '', [], [], [], 1, 1);
inq = q;
tic;
[x_ashIterator, status, history, T_nt_hist] = SigmoidLogBarrierSolver( Para, [], [], [], '', [], [], 0);

toc
t(1,1) = toc;
if any(S*x_ashIterator+q > 0 )
	disp('Inequality not satisfied');
    max( S*x_ashIterator+q)
end

%toc(ashIteratorTime)

%return;
% generate VRml
%    PlanePara = reshape(x_ashIterator,3,[]);
%    FitDepthPPCP = FarestDist*ones(1,55*305);
%    FitDepthPPCP(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded ))).*Ray(:,~maskSkyEroded ),1))';
%    FitDepthPPCP = reshape(FitDepthPPCP,55,[]);
%    [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
%    Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
%          Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
%          RR =permute(Ray,[2 3 1]);
%          temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
%          WrlFacestHroiReduce(Position3DFitedPPCP,PositionTex,SupOri, [ filename],[ filename 'AshuMinimizer'], ...
%                              [ OutPutFolder '/'], 0, 0);
%    [OutPutFolder '/' filename 'AshuMinimizer']

%x_ashIterator = Ash_L1_Minimizer(A,b);

% solve here for x using another method, and compare outputs
% solve by sedumi

% A = [An1; An2; An3; An4];
% clear An1 An2 An3 An4;

% =============================

% sedumiTime = tic;
tic
xsedumi = sdpvar( size(A,2), 1);
opt = sdpsettings('solver','sedumi','cachesolvers',1, 'verbose', 0);
obj = norm(A*xsedumi-b, 1);
F = set(S*xsedumi+q<=0);
%[model,recoverymodel] = export(F,obj,opt);

sol = solvesdp(F, norm(A*xsedumi-b, 1),opt);
%[x,y] = sedumi(model.A,model.b,model.C,model.K);
%assign(recover(recoverymodel.used_variables),y);
xsedumi = double(xsedumi);
toc
t(2,1) = toc;
% toc(sedumiTime)

% generate VRml
%    PlanePara = reshape(xsedumi,3,[]);
%    FitDepthPPCP = FarestDist*ones(1,55*305);
%    FitDepthPPCP(~maskSkyEroded) = (1./sum(PlanePara(:,Sup2Para(SupEpand(~maskSkyEroded ))).*Ray(:,~maskSkyEroded ),1))';
%    FitDepthPPCP = reshape(FitDepthPPCP,55,[]);
%    [Position3DFitedPPCP] = im_cr2w_cr(FitDepthPPCP,permute(Ray,[2 3 1]));
%    Position3DFitedPPCP(3,:) = -Position3DFitedPPCP(3,:);
%          Position3DFitedPPCP = permute(Position3DFitedPPCP,[2 3 1]);
%          RR =permute(Ray,[2 3 1]);
%          temp = RR(:,:,1:2)./repmat(RR(:,:,3),[1 1 2]);
%          WrlFacestHroiReduce(Position3DFitedPPCP,PositionTex,SupOri, [ filename],[ filename 'sedumi'], ...
%                              [ OutPutFolder '/'], 0, 0);
%    [OutPutFolder '/' filename 'sedumi']
   
norm(A*x_ashIterator-b,1)
norm(A*xsedumi-b,1)
plot(x_ashIterator,'r');
hold on;
plot(xsedumi,'g');

norm(x_ashIterator-xsedumi,1) / norm(xsedumi,1)
norm(A*x_ashIterator-b,1)
norm(A*(x_ashIterator-xsedumi),1) / norm(A*x_ashIterator-b,1)
