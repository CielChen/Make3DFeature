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
function [w, alfa, status, history, T_nt_hist] = ...
          SigmoidSolver( Para, t_0, alpha_0, w_0, method, ptol, pmaxi, VERBOSE)
%
% l1 penalty approximate Problem with inequality constrain Solver
%
% exact problems form:
%
% minimize norm( Aw - b, 1) 
%
% sigmoid approximate form:
%
% minimize sum_i (1/alfa)*( log( 1+exp(-alfa*( A(i,:)*w - b(i)))) + ...
%                      log( 1+exp(alfa*( A(i,:)*w - b(i))))  )
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

global A b 


% FEASIBLAE DATA CHECK
[m,n]   = size(A);          % problem size: m examples, n features

%------------------------------------------------------------
%       INITIALIZE
%------------------------------------------------------------

% SIGMOID APPROXIMATION
NUMERICAL_LIMIT_EXP = 3e2;      % this value is calculated such that exp(2*NUMERICAL_LIMIT_EXP) < inf
MU_alpha = 1;  ;%1.1;  %1.5;   % for sigmoidal
MU_alpha2 = 1.3;  %5;%1.5;  %1.5;   % for sigmoidal
ALPHA_MAX = 50000;
if(isempty(alpha_0 )) alpha_0 = 1e-1; end	% ERROR, actually, it should  be related to min(A*x-b) is set later in the code. See below
ALPHA_MIN = 5000.0;		% should be atleast 100 -- ERROR

% NEWTON PARAMETERS
MAX_TNT_ITER    = 1;   %100;  %25;      % maximum Newton iteration
%ABSTOL          = 1e-8;     % terminates when the norm of gradient < ABSTOL
EPSILON         = 1e-6;  %1e-6;     % terminate when lambdasqr_by_2 < EPSILON

% LINE SEARCH PARAMETERS
ALPHA_LineSearch= 0.01;     % minimum fraction of decrease in norm(gradient)
BETA            = 0.5;      % stepsize decrease factor
MAX_LS_ITER     = 25;      % maximum backtracking line search iteration
%MIN_STEP_SIZE   = 1e-20;


% VARIABLE INITIALZE
w = zeros(size(A,2),1);

%save initial_w.mat w
% load initial_w.mat

% setting starting alpha_0
alpha_0 = 1 / max( abs(A*w-b) );
alfa = alpha_0;
alfa_max = alfa;


% -----------------------------------------------------------
% META LOOP FOR CHANGING t and ALPHA
% ----------------------------------------------------------
history=[];
T_nt_hist=[];
	
if VERBOSE
	disp(sprintf('%15s %15s %11s %12s %10s %10s %7s %15s %15s',...
    			'exact obj', 'primal obj', 'alpha', 'MlogExpTerm',...
		      'norm(g)', 'lambda^2By2', ... 
		'gap', 'search_step', 'newton_steps'));
			%'normg_sigmoid', 'normg_t', ...
end

status = -1;
%for outer_iter = 1:MAX_OUTER_ITER
%history_Inner = [];
% Aw_b = ;
logExpTerm = alfa./MU_alpha2*(A*w-b)';
expTerm = exp(logExpTerm);
dw =  zeros(n,1); % dw newton step
while alfa < ALPHA_MIN

%   history_Inner = [];

   

%  Makes the profiler not work!
%  if (VERBOSE >= 2)
%     disp(sprintf('%s %15s %15s %11s %12s %10s %10s %10s %10s %9s %6s %6s %10s %10s %10s',...
%     'iter', 'exact obj', 'primal obj', 'alpha', 'MlogExpTerm',...
%     'In_step', 'stepsize','norm(g)', 'norm(dw)', ...
%     'lambda^2By2','normg_sigmoid','normg_t', ...
%		'normdw_t', 'normdw_sigmoid'));
%  end

   %------------------------------------------------------------
   %               MAIN LOOP
   %------------------------------------------------------------
   %total_linesearch = 0;
   %ntiter = 0;
   %initialize Newton step
        
        logExpTerm = logExpTerm.*MU_alpha2;
        expTerm = expTerm.^MU_alpha2;
        %expTermNeg = exp(-logExpTerm);
        expTerm_Rec = 1./(1+expTerm);

        expTermNeg_Rec = 1./(1+exp(-logExpTerm));
        g_sigmoid = (expTermNeg_Rec - expTerm_Rec);
        gradphi = (g_sigmoid*A)'; %A1endPt) + ...
%        gradphi = gradphi_sigmoid;
   
%    newtonLoop = (ntiter <= MAX_TNT_ITER);
% 	while newtonLoop
        %ntiter = ntiter + 1;

        indicesInf = find( (logExpTerm > NUMERICAL_LIMIT_EXP) | (logExpTerm < -NUMERICAL_LIMIT_EXP) );
        %indicesOkay = setdiff(1:length(logExpTerm), [indicesInf_Pos, indicesInf_Neg]);
    
        h_sigmoid = expTerm .* (expTerm_Rec.^2);
        h_sigmoid(indicesInf) = 0;
     
        
        %------------------------------------------------------------
        %       CALCULATE NEWTON STEP
        %------------------------------------------------------------
         
        hessphi =  (sparse(1:m,1:m,h_sigmoid)*A)' * A;
        %       hessphi_sigmoid =  (h_sigmoid*(A.^2))';     %diagonal element only
            
            %	if DEBUG_FLAG && (any(h_sigmoid<0) || any(h_t<0) )
            %          disp(sprintf('h negative = %5.4e; h_t negative = %5.4e',h_sigmoid(h_sigmoid<0), h_t(h_t<0)));
            %       end
       
           
%       hessphi_t = (h_t*(S.^2))' / (2*alfa);  %Diagonal Element only
            
        %hessphi = hessphi_sigmoid;  %HACK
       
        dw = - hessphi \ gradphi;
                     
%             dw = - (1./hessphi) .* gradphi;       % Diagonal Hessian
             dw = dw / (2*alfa);
       
       % newton decrement===========================================
%       lambdasqr = full(-( gradphi'*dw) );
       %lambdasqr2 = full(dw'*(hessphi+ hessphi_t)'*dw);
       % ===========================================================
	
       
       %------------------------------------------------------------
       %   BACKTRACKING LINE SEARCH
       %------------------------------------------------------------
	s = 1;
       normg = norm(gradphi);
       lsiter = 0;
       backIterationLoop = true;
       while backIterationLoop
           lsiter = lsiter+1;
           new_w = w+s*dw;
%            Aw_b = (A*new_w-b);
           logExpTerm = alfa*(A*new_w-b)';
           expTerm = exp(logExpTerm);
       
            expTerm_Rec = 1./(1+expTerm);
            expTermNeg_Rec = 1./(1+exp(-logExpTerm));
            g_sigmoid = (expTermNeg_Rec - expTerm_Rec);
            gradphi = (g_sigmoid*A)'; %A1endPt) + ...
            %gradphi = gradphi_sigmoid;  
           
            backIterationLoop = (lsiter <= MAX_LS_ITER) && ( norm(gradphi) > (1-ALPHA_LineSearch*s)*normg);
            s = BETA*s;
        end

%        total_linesearch = total_linesearch + lsiter;

%       if VERBOSE >= 2
%       		Exact_phi_sigmoid = norm( logExpTerm/alfa, 1); % only for debug
%       		Exact_f = Exact_phi_sigmoid + logBarrierValue / t;	% MIN -- fixed
%		%normdw = norm(dw);
%		normg_sigmoid = norm(gradphi_sigmoid);
%		normg_t = norm(gradphi_t);
%       		normdw_sigmoid = 0;%norm( dw_sigmoid);
%       		normdw_t = 0;%norm( dw_t);
%       		normdw = norm( dw);
%       		f_new = phi_sigmoid + logBarrierValue / t;	% MIN -- fixed ? 
%          disp(sprintf('%4d %15.6e %15.6e %4.5e %10.2e %10.2e %10.2e %6d %6d %10.2e %6d %6d %6d %6d',...
%                ntiter,Exact_f, f_new, alfa, max(abs( logExpTerm)), s0, s, normg, normdw, lambdasqr/2, ...
%					normg_sigmoid, normg_t, normdw_t, normdw_sigmoid));
%       end

       %------------------------------------------------------------
       %   STOPPING CRITERION
       %------------------------------------------------------------
   
%       if (lambdasqr/2 <= EPSILON)
%            status = 1;
%            %disp('Minimal Newton decrement reached');
%       end

%       newtonLoop = (ntiter <= MAX_TNT_ITER) && (lsiter < MAX_LS_ITER);  %
       
       % set new w 
       w = new_w;
       %alfa = min(alfa*MU_alpha, alfa_max);
       % If you want to do the above, i.e., increase the value in each
       % Newton step, uncomment the gradient calculation, if ntiter == 1

%    end % -----------END of the MAIN LOOP-----------------------------

   % Tighten the sigmoid and log approximation

    
% 	if VERBOSE
%        		Exact_phi_sigmoid = norm( logExpTerm/alfa, 1); % only for debug
% 
% 		normg_sigmoid = norm(gradphi_sigmoid);
%        		normdw = norm( dw);
%        		normg = norm( gradphi);
%             expTermNeg = exp(-logExpTerm);
%              phi_sigmoid = ( log( 1+ expTermNeg ) + log( 1+ expTerm) );
%         phi_sigmoid( indicesInf_Pos ) = logExpTerm( indicesInf_Pos ); 
%         phi_sigmoid( indicesInf_Neg ) = -logExpTerm( indicesInf_Neg ); 
%         phi_sigmoid = (1/alfa) * sum(phi_sigmoid,2);
%  
%        		f_new = phi_sigmoid;	% MIN -- fixed ? 
%             Exact_f = Exact_phi_sigmoid;	% MIN -- fixed
%           disp(sprintf('%15.6e %15.6e %4.5e %10.2e %10.2e %10.2e %10.2d %10d %10d',...
%                 Exact_f, f_new, alfa, max(abs( logExpTerm)), normg, lambdasqr/2, ...
% 				gap, total_linesearch, ntiter));
% 				%	normg_sigmoid, normg_t, ...
% 	end

%   T_nt_hist = [T_nt_hist history_Inner];
%   history=[history [length(history_Inner)]];
   
	%if alfa >= ALPHA_MIN/MU_alpha
	%else
	%	epsilon_gap = (1-alfa/ALPHA_MIN)*1e2 + (alfa/ALPHA_MIN)*EPSILON_GAP; 
    %end

    
%   alfa = min(alfa*MU_alpha, ALPHA_MAX);
%    alfa_max = min(alfa_max*MU_alpha2, ALPHA_MAX);
%    if (alfa > ALPHA_MIN) && (status >= 1)
%        t = MU_t*t;         % if just waiting for gap, then increase it at double speed
%    end

    alfa = alfa*MU_alpha2;
%    alfa_max = min(alfa*MU_alpha2, ALPHA_MAX);
    
end

return;

