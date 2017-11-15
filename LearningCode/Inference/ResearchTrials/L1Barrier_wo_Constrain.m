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
function [x,status,history] = L1Barrier_wo_Constrain(Para,method,ptol,pmaxi, VERBOSE)
%function [x,status,history] = L1Barrier_wo_Constrain(A,b,t_0,method,ptol,pmaxi)
%
% Fast L1 - norm Solver
%
% L1 - norm Solver Solves problems of the following form:
%
% minimize | A*x - b|L1
% 
% where variable is x and problem data are A and b.
%
% INPUT
%
%  A       : mxn matrix; input data. each column corresponds to each feature
%  b       : m vector; class label
%
%  method  : string; search direction method type
%               'cg'   : conjugate gradients method, 'pcg'
%               'pcg'  : preconditioned conjugate gradients method
%               'exact': exact method (default value)
%  ptol    : scalar; pcg relative tolerance. if empty, use adaptive rule.
%  pmaxi   : scalar: pcg maximum iteration. if empty, use default value (500).
%
% OUTPUT
%
%  x       : n vector; 
%  status  : scalar; +1: success, -1: maxiter exceeded
%  history :
%            row 1) phi
%            row 2) norm(gradient of phi)
%            row 3) cumulative cg iterations
%
% USAGE EXAMPLE
%
%  [x,status] = l2_logreg(A,b,lambda,'pcg');
%

% Written by Kwangmoo Koh <deneb1@stanford.edu>
% adopted by Min Sun

%------------------------------------------------------------
%       INITIALIZE
%------------------------------------------------------------
global A D p b;

% LOG BARRIER METHOD
MAX_LOGB_ITER = 100;
EPSILON_GAP = 2e-4;
MU_t = 100;   % for t -- log barrier.  Changed ASH
%if(isempty(t_0)) t_0 = 1; end
t_0 = 1;
t = t_0;


% NEWTON PARAMETERS
MAX_TNT_ITER    = 100;      % maximum (truncated) Newton iteration
ABSTOL          = 1e-8;     % terminates when the norm of gradient < ABSTOL
EPSILON         = 1e-7;     % terminate when lambdasqr_by_2 < EPSILON
StopNorm        = 0;        % set to 0 using newton decrement

% LINE SEARCH PARAMETERS
ALPHA           = 0.01;     % minimum fraction of decrease in norm(gradient)
BETA            = 0.5;      % stepsize decrease factor
MAX_LS_ITER     = 50;      % maximum backtracking line search iteration
Eps             = -1e-50;     % gap for inequality
normg_Flag      = 1;         % if evaluate function set normg_Flag = 1

[m,n]   = size(A);          % problem size: m examples, n features
A2 = A.^2;


%if(isempty(pmaxi)) pcgmaxi = 500; else pcgmaxi = pmaxi; end
%if(isempty(ptol )) pcgtol = 1e-4; else pcgtol  = ptol;  end
pcgmaxi = 500;
pcgtol = 1e-4;

% INITIALIZE
pobj  = Inf; s = inf; pitr  = 0 ; pflg  = 0 ; prelres = 0; pcgiter = 0;
history = [];
status = -1;
% feasible starting point


x = zeros(n,1); dx =  zeros(n,1);
y = max( abs( A*x-b))-Eps- (-1e-2); dy =  zeros(m,1);


% check is x y feasible start
% if max(A*(x) - (y)-b)< 0 && max(-A*(x) - (y)+b) < 0
%    disp('Feasible start');
% end


%------------------------------------------------------------
%      LOG BARRIER OUTER LOOP
%------------------------------------------------------------
%for lbiter = 1:MAX_LOGB_ITER
%while true
% return;
%end
for LogBIter = 1:MAX_LOGB_ITER
if VERBOSE
 disp(sprintf('%s %15s %10s %10s %10s %s %s %6s %10s %6s',...
     'iter','primal obj','stepsize','norg(g)','lambdasqr','LSiter', 'LSiter_fea''p_flg','p_res','p_itr'));
end
    %------------------------------------------------------------
    %               MAIN LOOP
    %------------------------------------------------------------

    status = -1; % initalized to -1;
%    nt_hist = [];
        Ax_b = A*x - b;
        Ax_b_y = Ax_b + y;
        Neg_Ax_b_y = -Ax_b + y;
        g_1 = (1./Neg_Ax_b_y - 1./Ax_b_y);
        g_2 = (t - 1./Ax_b_y - 1./Neg_Ax_b_y);
        gradphi_x = (g_1'*A)';
    for ntiter = 0:MAX_TNT_ITER
        D_1 = (1./Neg_Ax_b_y).^2;
        D_2 = (1./Ax_b_y).^2;
        D = 2./(y.^2 + Ax_b.^2 );
        g = g_1 +(D_1-D_2).*(1./(D_1+D_2)).*g_2;
        gradphi_x_eli = (g'*A)';

        %------------------------------------------------------------
        %       CALCULATE NEWTON STEP
        %------------------------------------------------------------
%         switch lower(method)
%             case 'pcg'
%             p  = 1./(A2'*D);
%             if (isempty(ptol)) pcgtol = min(0.1,norm(gradphi_x_eli)); end
%                 [dx, pflg, prelres, pitr, presvec] = ...
%                 pcg(@AXfunc,-gradphi_x_eli,pcgtol,pcgmaxi,@Mfunc,[],[]);
% %                A,D,[],1./(A2'*D));
%             if (pitr == 0) pitr = pcgmaxi; end
% 
%             case 'cg'
%             if (isempty(ptol)) pcgtol = min(0.1,norm(gradphi_x_eli)); end
%                 [dx, pflg, prelres, pitr, presvec] = ...
%                 pcg(@AXfunc,-gradphi_x_eli,pcgtol,pcgmaxi,[],[],[]);
% %                A,D,[],[]);
%             if (pitr == 0) pitr = pcgmaxi; end
% 
%             otherwise % exact method                
 %               hessphi_x = A'*sparse(1:m,1:m,D)*A;
                hessphi_x =  (sparse(1:m,1:m, D)*A)' * A;
                dx = -hessphi_x\gradphi_x_eli;
%          end
         dy = (1./(D_1 + D_2)) .*(-g_2 + (D_1 - D_2).*(A*dx));
         %pcgiter = pcgiter+pitr;

        % function value and normg for back tracking line search or stoping critera
        normg = norm([gradphi_x; g_2]);

        phi = sum(y) - sum( log( [Ax_b_y; Neg_Ax_b_y]))/t;
        lambda_sqr = -gradphi_x'*dx - g_2'*dy;
        %------------------------------------------------------------
        %   BACKTRACKING LINE SEARCH
        %------------------------------------------------------------
        s = 1;
%      if false % debug
        Delta = A*(dx) - (dy);
        Delta_Negdy = -A*(dx) - (dy);
        LSiter = 0;
        while any( (s*Delta - Neg_Ax_b_y) >= Eps) || any(( s*Delta_Negdy - Ax_b_y) >= Eps)
            s = BETA*s;
            LSiter = LSiter + 1;
        end
        for lsiter = 1:MAX_LS_ITER
            new_x = x + s*dx;
            new_y = y + s*dy;
            Ax_b = A*new_x - b;
            Ax_b_y = Ax_b + new_y;
            Neg_Ax_b_y = -Ax_b + new_y;
            
%             if normg_Flag
               g_1 = (1./Neg_Ax_b_y - 1./Ax_b_y);
               g_2 = (t - 1./Ax_b_y - 1./Neg_Ax_b_y);
               gradphi_x = (g_1'*A)';
               if (norm([gradphi_x; g_2])<=(1-ALPHA*s)*normg) break; end
               s = BETA*s;
%             else
%                % evaluate function value
%                new_phi = sum(y) - sum( log( [Anew_x_b_y; Neg_Anew_x_b_y]))/t;
%                if new_phi <= phi +ALPHA*s*[gradphi_x; g_2]'*[dx; dy] break; end
%                s = BETA*s;
%             end
        end
%      end
        x = new_x;
%         dx;
%         x';
        y = new_y;
%        if VERBOSE
%	disp(sprintf('%4d %15.6e %10.2e %10.2e %10.2e %4d %3d %6d %10.2e %6d',...
%                ntiter,phi,s,normg, lambda_sqr/2, lsiter, LSiter, pflg,prelres,pitr));
%        nt_hist = [nt_hist [phi; normg; pcgiter]];
%	end
        %------------------------------------------------------------
        %   STOPPING CRITERION
        %------------------------------------------------------------
        if (lsiter == MAX_LS_ITER) disp('MaxLSIter'); break; end
      if StopNorm
        if (normg < ABSTOL) 
           status = 1;
           %disp('Absolute normg tolerance reached.');
%           disp(sprintf('%d/%d',sum(abs((A2'*h)./(2*lambda))<0.5),n));
           break;
        end
      else
        if (lambda_sqr/2 <= EPSILON)
            status = 1;
            %disp('Absolute Lambda tolerance reached.');
            break;
        end
      end
    end
    if status == -1 disp('Error status -1'); end
    
    %--------------  decreasing the gap -------------
    gap = m/t;
	disp(gap);
    %history=[history [length(nt_hist); gap]];
    if gap< EPSILON_GAP break; end
    t = MU_t*t;
    
end

return;


%------------------------------------------------------------
%   CALL BACK FUNCTIONS FOR PCG
%------------------------------------------------------------
function y = AXfunc(x)
    global A D;
    y = A'*(D.*(A*x));

function y = Mfunc(x)
    global p;
    y = x.*p;
