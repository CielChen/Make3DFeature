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
function [w, alpha, status, history] = Min_minimizer( Para, t, alpha, w, A, b, S, q, method, ptol, pmaxi, mu, VERBOSE, LogBarrierFlg)
%
% l1 penalty approximate Problem with inequality constrain Solver
%
% exact problems form:
%
% minimize norm( Aw - b, 1) st, Sw<=q
%
% sigmoid approximate form:
%
% minimize sum_i (1/alpha)*( log( 1+exp(-alpha*( A(i,:)*w - b(i)))) + ...
%                      log( 1+exp(alpha*( A(i,:)*w - b(i))))  )
%           st, Sw <= q (log barrier approximation)
% 
% where variable is w and problem data are A, b, S, and q.
%
% INPUT
%
%  A       : mxn matrix;
%  b       : m vector; 
%  S       : lxn matrix;
%  q       : l vector;
%
%  method  : string; search direction method type
%               'cg'   : conjugate gradients method, 'pcg'
%               'pcg'  : preconditioned conjugate gradients method
%               'exact': exact method (default value)
%  ptol    : scalar; pcg relative tolerance. if empty, use adaptive rule.
%  pmaxi   : scalar: pcg maximum iteration. if empty, use default value (500).a
%  mu      : approximate_closeness factor; bigger -> more accurate
%  VERBOSE : enable disp  
%
% OUTPUT
%
%  w       : n vector; classifier
%  status  : scalar; +1: success, -1: maxiter exceeded
%  history :
%            row 1) phi (objective value)
%            row 2) norm(gradient of phi)
%            row 3) cumulative cg iterations
%
% USAGE EXAMPLE
%
%  [w,status] = (A, b, S, q, 'pcg');
%

% template by Kwangmoo Koh <deneb1@stanford.edu>
% Modified by Min Sun <aliensun@stanford.edu>
%------------------------------------------------------------
%       INITIALIZE
%------------------------------------------------------------

% NEWTON PARAMETERS
MAX_TNT_ITER    = 200;      % maximum (truncated) Newton iteration
ABSTOL          = 1e-8;     % terminates when the norm of gradient < ABSTOL
EPSILON         = 1e0;  %1e-6;     % terminate when lambdasqr_by_2 < EPSILON

% LINE SEARCH PARAMETERS
ALPHA_LineSearch= 0.01;     % minimum fraction of decrease in norm(gradient)
BETA            = 0.5;      % stepsize decrease factor
MAX_LS_ITER     = 100;      % maximum backtracking line search iteration

if(isempty(pmaxi)) pcgmaxi = 500; else pcgmaxi = pmaxi; end
if(isempty(ptol )) pcgtol = 1e-4; else pcgtol  = ptol;  end

% sigmoid approximation
ALPHA_MAX = 500000;
Eps = 0;
if(isempty(alpha )) alpha = 1; end
%MIN_ALPHA = 10000;
mu = 1;
%if(isempty(mu)) mu = 1; end % how to tune mu so that norm(gradient) decrease???

% log barrier moethd
h_t_thre = 1e15;
%t = 1;
%mu_t = 1.5;

% Data related Parameter
[m,n]   = size(A);          % problem size: m examples, n features
[l,n_S]   = size(S);

if ~isempty(S)
 if n_S ~= n; disp('size inconsistance'); return; end  % stop if matrix dimension mismatch
end

% INITIALIZE
pobj  = Inf; s = Inf; pitr  = 0 ; pflg  = 0 ; prelres = 0; pcgiter = 0;
history = [];

%w0 = ones(n,1); 
% find strickly feasible starting w
%opt = sdpsettings('solver','sedumi','cachesolvers',1, 'verbose', 0);
%w = sdpvar(n,1);
%Strick_feasible_gap = (1/Para.ClosestDist -1/Para.FarestDist)/4*ones(size(q));
%Strick_feasible_gap(q == 0) = 1;
%sol = solvesdp(set(S*w+q + Strick_feasible_gap<=0),norm(w0 - w),opt);
%w = double(w);
dw =  zeros(n,1); % dw newton step

if VERBOSE
   disp(sprintf('%s %15s %15s %11s %12s %6s %10s %6s %6s %6s %10s',...
    'iter', 'exact obj', 'primal obj', 'alpha', 'MlogExpTerm', 'stepsize','norm(g)', 'lambda^2By2','p_flg','p_res','p_itr'));
end

%------------------------------------------------------------
%               MAIN LOOP
%------------------------------------------------------------

for ntiter = 0:MAX_TNT_ITER

    % gradient related
    logExpTerm = alpha*(A*w-b);
    expTerm = exp(logExpTerm);
    expTermNeg = exp(-logExpTerm);
    g = sparse( 1./(1+expTermNeg) - 1./(1+expTerm) )*t;
    if LogBarrierFlg
       g_t = sparse(-1./(S*w+q));% log barrier  
    end

     % ======= origin of bad condition of Hessian
     h = exp( logExpTerm - 2*log(1+expTerm) )*t;
     h_cond = sum( h(1:Para.A1endPt) <=Eps ) >0 &&...
              sum( h( (Para.A1endPt+1):(Para.A2endPt)) <=Eps) >0 &&...
              sum( h( (Para.A2endPt+1):(Para.A3endPt)) <=Eps) >0 &&...
              sum( h( (Para.A3endPt+1):end) <=Eps) >0;
     if h_cond % heuristic of condition of hessphi (usually stop 1 to 5 step earlier)
        if VERBOSE
%           disp('bad condition');
        end
%         break;        
     end 
     % ==========================================
    if LogBarrierFlg
       h_t = 1./(S*w+q).^2;% log barrier
    end
     if any(h_t >= h_t_thre) % heuristic of condition of hessphi (usually stop 1 to 5 step earlier)
        if VERBOSE
%           disp('g_t bad condition');
        end
%         break;        
     end

%    gradphi = A'*g;
    gradphi = A( 1:Para.A1endPt,:)'*g(1:Para.A1endPt) + ...
        A( (1+Para.A1endPt):Para.A2endPt,:)'*g(( Para.A1endPt+1):Para.A2endPt) + ...
        A( (1+Para.A2endPt):Para.A3endPt,:)'*g( (Para.A2endPt+1):Para.A3endPt) + ...
        A( (1+Para.A3endPt):end,:)'*g( (Para.A3endPt+1):end);
    if LogBarrierFlg
       gradphi_t = S'* g_t;
    end
    normg = norm(gradphi+gradphi_t);

    phi = sum( (1/alpha)*( log( 1+ expTermNeg) + log( 1+ expTerm)), 1);
    Exact_phi = norm( logExpTerm/alpha, 1); % only for debug
    logBarrierValue = -sum(log( -S*w-q));
    %------------------------------------------------------------
    %       CALCULATE NEWTON STEP
    %------------------------------------------------------------
    switch lower(method)
        case 'pcg'
%        if (isempty(ptol)) pcgtol = min(0.1,norm(gradphi)); end
%        [dw, pflg, prelres, pitr, presvec] = ...
%            pcg(@AXfunc,-gradphi,pcgtol,pcgmaxi,@Mfunc,[],[],...
%                A,h,2*lambda,1./(A2'*h+2*lambda));
%        if (pitr == 0) pitr = pcgmaxi; end

        case 'cg'
%        if (isempty(ptol)) pcgtol = min(0.1,norm(gradphi)); end
%        [dw, pflg, prelres, pitr, presvec] = ...
%            pcg(@AXfunc,-gradphi,pcgtol,pcgmaxi,[],[],[],...
%                A,h,2*lambda,[]);
%        if (pitr == 0) pitr = pcgmaxi; end

        otherwise % exact method
%        hessphi = A'*sparse(1:m,1:m,h)*A;
        hessphi = ...
          ( A( 1:Para.A1endPt,:)' *...
            sparse(1:Para.A1endPt, 1:Para.A1endPt, h(1:Para.A1endPt)) * ...
            A( 1:Para.A1endPt,:) + ...
            A( (1+Para.A1endPt):Para.A2endPt,:)' *...
            sparse(1:(Para.A2endPt-Para.A1endPt), 1:(Para.A2endPt-Para.A1endPt), h( (Para.A1endPt+1):(Para.A2endPt))) *...
            A( (1+Para.A1endPt):Para.A2endPt,:) + ...
            A( (1+Para.A2endPt):Para.A3endPt,:)' *...
            sparse(1:(Para.A3endPt-Para.A2endPt), 1:(Para.A3endPt-Para.A2endPt), h( (Para.A2endPt+1):(Para.A3endPt))) *...
            A( (1+Para.A2endPt):Para.A3endPt,:) + ...
            A( (1+Para.A3endPt):end,:)' *...
            sparse(1:(Para.A4endPt-Para.A3endPt), 1:(Para.A4endPt-Para.A3endPt), h( (Para.A3endPt+1):end)) *...
            A( (1+Para.A3endPt):end,:)) *...
            (2*alpha);
        if LogBarrierFlg 
           hessphi_t = S'*sparse(1:length(h_t), 1:length(h_t), h_t)*S;
%           if VERBOSE
 %             disp('condition number of H');
 %             condest((hessphi+ hessphi_t))
 %          end
           dw = -(hessphi+ hessphi_t)\(gradphi + gradphi_t);
        else
           dw = -hessphi\gradphi;
        end
    end

    % newton decrement===========================================
    %lambdasqr = full(-(gradphi + gradphi_t)'*dw);
    lambdasqr = full(dw'*(hessphi+ hessphi_t)'*dw);
    % ===========================================================
    
    if VERBOSE
       disp(sprintf('%4d %15.6e %15.6e %4.5e %10.2e %10.2e %6d %10.2e %6d',...
                ntiter,Exact_phi, phi, alpha, max(abs( logExpTerm)), s,normg, lambdasqr/2,pflg,prelres,pitr));
    end

    history = [history [Exact_phi + logBarrierValue; phi+logBarrierValue; ...
                        alpha; max(abs( logExpTerm)); normg; lambdasqr/2; pcgiter]];
    %------------------------------------------------------------
    %   STOPPING CRITERION
    %------------------------------------------------------------
    if ( normg < ABSTOL) 
        status = 1;
        disp('Minimal norm( gradient) reached.');
%        disp(sprintf('%d/%d',sum(abs((A2'*h)./(2*lambda))<0.5),n));
        return;
    end
    if (lambdasqr/2 <= EPSILON)
        status = 1;
        disp('Minimal Newton decrement reached');
        return;
    end
    % hacking stop
    if ntiter ~=0
       if abs(history(6,end-1) - history(6,end)) < EPSILON
          disp('lambdasqr not improving');
          break;
       end
    end

    pcgiter = pcgiter+pitr;
    %------------------------------------------------------------
    %   BACKTRACKING LINE SEARCH
    %------------------------------------------------------------
    s = 1;
    while max( S*(w+s*dw)+q) > 0 s = BETA*s; end % first set the new w inside
    for lsiter = 1:MAX_LS_ITER
        new_w = w+s*dw;
        logExpTerm = alpha*(A*new_w-b);
        expTerm = exp(logExpTerm);
        expTermNeg = exp(-logExpTerm);
        g = sparse( 1./(1+expTermNeg) - 1./(1+expTerm) );
        newgradphi = A'*g;
        if (norm(newgradphi)<=(1-ALPHA_LineSearch*s)*normg) break; end
        s = BETA*s;
    end
    if (lsiter == MAX_LS_ITER) disp('Max lineSearch iteration reached'); break; end
    w = new_w;

    % check if new_w feasible
    if ~all(S*w+q<=0)
       disp('new_w infeasible'); 
       break;
    end
    % tighten the sigmoid approximation
    alpha = min(alpha*mu, ALPHA_MAX);

    % tighten the log barrier approximation
    %t = t*mu_t;
end
status = -1;

%------------------------------------------------------------
%   CALL BACK FUNCTIONS FOR PCG
%------------------------------------------------------------
%function y = AXfunc(x,A,h,d,p)
%    y = A'*(h.*(A*x))+d.*x;

%function y = Mfunc(x,A,h,d,p)
%    y = x.*p;
return;

