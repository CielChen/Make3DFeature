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
function [x, fail, info] = Ash_L1_Minimizer(A, b, tol, VERBOSE)
% [x, fail] = function Ash_L1_Minimizer(A, b, tol, VERBOSE)
% Solves:      x = minimize_x || Ax-b ||_1
%              x \in Re^{Nx1},   A \in Re^{MxN},   b \in Re^{Mx1}
% uses sigmoidal approximation to the gradient,
% fail:    returns 1 if optimization fails.
% info:    detailed results of optimization
% tol:      tolerance
% VERBOSE:  false by default
%
% Optimizatio parameters to be tweaked inside the function are;
% ALPHA_MAX, alpha_init, mu, t0, beta

fail = 0;

if nargin < 2
    disp('Type help Ash_L1_Minimizer');
    fail = 1;
    return;
end
if size(A,1) ~= size(b,1)
    disp('Vectors of different size. Type help Ash_L1_Minimizer');
    fail = 1;
    return;
end
if size(A,1) < size(A,2)
    disp('Multiple solutions possible');
end

if nargin < 3
    tol = 1e-12;         % should be tweaked.
end
if nargin < 4
    VERBOSE = false;
end

%load data_test_L1.mat

%===Magic Numbers
alpha_init = 1;
mu = 1.01;
ALPHA_MAX = 50000;
MIN_ALPHA = 5000;
MAX_ITER = 1000;  %1000;
x_init = zeros(size(A,2),1);

%initialize loop
alpha = alpha_init;
x = x_init;
info.numIter = 0;

[obj, g, H, actual] = f_Hessian_evaluate(A,b,x,alpha);
d = - H \ g;    % the Newton direction
info.newtonDecrement = d'*H*d;      % could also be g'*d

% start iteration loop
%while g'*d > tol && numIter < MAX_ITER && alpha < MIN_ALPHA
while info.newtonDecrement > tol && info.numIter < MAX_ITER && alpha < MIN_ALPHA
	t = find_t_byBacktracking(d, g, A, b, x, alpha);
	x = x + t * d;
	[obj, g, H, actual, IllConditionStop] = f_Hessian_evaluate(A,b,x,alpha);
    if IllConditionStop
        disp('Bad conditioning of Hessian....');
        break;
    end
  	d = - H \ g;	% Newton step; probably will be faster, if banded
	alpha = min(alpha*mu, ALPHA_MAX);		% make the approximation more tight
    
    info.newtonDecrement = d'*H*d;      % could also be g'*d
    if VERBOSE
        disp(['Newton decrement. lambda^2 = d^THd = ' num2str(info.newtonDecrement)]);
        disp(['Iteration number: ' num2str(numIter)]);
        disp(['Alpha:     ' num2str(alpha)]);
        disp(['Approximate ||Ax-b||_1 value = ' num2str(obj) ',    Exact value = ' num2str(actual)]);
    end
    info.numIter = info.numIter+1;
end

if info.numIter >= MAX_ITER && info.newtonDecrement > tol
    disp('Error: Maximum Iterations reached or newtonDecrement is high or NaN somewhere.');
    fail = 1;
end

info.alpha = alpha;
info.IllConditionStop = IllConditionStop;

return;



%============ Function that evaluates the Hessian and the gradient ========
function [obj, g, H, actual, stop] = f_Hessian_evaluate(A,b,x,alpha)
% alpha is the sigmoidal approximation

% objective
[obj, actual] = f_obj_evaluate(A, b, x, alpha);

logExpTerm = alpha*(A*x-b);
expTerm = exp(logExpTerm);
expTermNeg = exp(-logExpTerm);

%evaluating the Hessian
n = exp( logExpTerm - 2*log(1+expTerm) );
nonZeroN = find(n~=0);
stop = (length(find(n==0))/length(n)) > 0;
%H = 2*alpha* A(nonZeroN,:)' * diag(n(nonZeroN)) * A(nonZeroN,:);
H = 2*alpha* A' * diag(n) * A;

% evaluating the gradient
m = 1./(1+expTermNeg) - 1./(1+expTerm);
%m = (expTerm-1) ./ (expTerm+1);
%g = A(nonZeroN,:)'*diag(m(nonZeroN))*ones(size(A(nonZeroN,:),1),1);
g = A'*diag(m)*ones(size(A,1),1);

return;


%====== Function that just evaluates ||Ax-b||_1 ========
function [obj, actual] = f_obj_evaluate(A, b, x, alpha)

expTerm = exp(alpha*(A*x-b));
expTermNeg = exp(-alpha*(A*x-b));

% Appromixate (smooth) objective value
obj = ones(size(A,1),1)'* (1/alpha)* ( log(1+expTermNeg) + log(1+expTerm) );

%actual value of ||Ax-b||_1
actual = norm(A*x-b,1);

return;



% =========== Function to calculate step size t by back-tracking line search ====	
function [t] = find_t_byBacktracking(d, g, A, b, x, alpha, beta, t0)

if nargin < 7
    beta = 0.5;     
    t0 = 1;         % can be made faster here
end

t = t0;
f_x = f_obj_evaluate(A,b,x,alpha);
gd = g'*d;

% Refer Convex Optimization, Boyd page 464
while f_obj_evaluate(A,b,x+t*d,alpha) > f_x + t*gd
    t = t*beta;
end

return;
