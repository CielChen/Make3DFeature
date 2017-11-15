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
function [x,y,flag]=warmStart(A1,A2,b,c,M1,M2,NuSupSize)

m=M1*2+M2;
t=1000;
epsilon=1e-5;
mu=20;
alpha=0.2;
beta=0.5;
%% may be later I can implement backtracking line search, as of now, working with small alpha;

x=ones(3*NuSupSize,1);
y=ones(M1,1);

while((m/t)>epsilon)
   goOn=boolean(1);
   while(goOn)
      %D1=sparse(diag(1./([t1].^2)));
      %D2=sparse(diag(1./([t2].^2)));
%       size(b)
%       size(A1)
%       size(x)
      t1=b-A1*x+y;
      it1=1./t1;
      it1s=1./(t1.^2);
      t2=-b+A1*x+y;
      it2=1./t2;
      it2s=1./(t2.^2);
      t3=c-A2*x;
      t4=(it1s-it2s);
      t5=(it1s+it2s);
      D=spdiags((2./([y.^2 + (b-A1*x).^2])),0,M1,M1);
      D3=spdiags(1./(t3.^2),0,M2,M2);
      g1=A1'*[it1-it2]-A2'*[1./t3];
      g2=t*ones(M1,1)-it1-it2;
      %g=g1+A1'*(D1-D2)*inv(D1+D2)*g2; 
      g=g1+A1'*[(t4./t5).*g2];
%       DeltaxNt=cgs(A1'*D*A1+A2'*D3*A2,-g);
%       DeltaxNt=pcg(A1'*D*A1+A2'*D3*A2,-g);%cgs(A1'*D*A1+A2'*D3*A2,-g);
      DeltaxNt=-pinv(A1'*D*A1+A2'*D3*A2)*g;
      DeltayNt=((t4.*(A1*DeltaxNt))-g2)./t5;
      x=x+alpha*DeltaxNt;
      y=y+alpha*DeltayNt;
      toler=norm(DeltaxNt./x)
      if(toler<epsilon)
        goOn=boolean(0);
      end
   end
   t=mu*t;
end
