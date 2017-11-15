/*
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
#include <stdio.h>
#include <vector>
#include "cv.h"
#include "mat.h"
#include <map>
//#include "mex.h"
using namespace std;

#define MAT_A prhs[0]
#define VECT_B prhs[1]
#define MAT_S prhs[2]
#define VECT_Q prhs[3]
#define VECT_W plhs[0]
#define DOUBLE_ALPHA plhs[1]
#define INT_STATUS plhs[2]
#define DEBUG 0
#define DEBUG_MATRIX 0
#define DEBUG_PERF_MON 1

map<CvMat *,vector<int> **> cached_row_nonzero_indices;
map<CvMat *,vector<int> **> cached_col_nonzero_indices;

void printMat(char*name, CvMat * mat);

CvMat* createCvMatFromMatlab(const mxArray *mxArray);

void cvSparseMatMul(CvMat *A, CvMat *B, CvMat *C, int tABC) {
    double sum;
    int i, j, k, m, n, l;
    vector<int> **A_i_nonzero_indices = 0;
    vector<int> **B_j_nonzero_indices = 0;
    
    if (tABC == CV_GEMM_A_T) {
        m = cvGetSize(A).width;
        n = cvGetSize(A).height;
    } else {
        m = cvGetSize(A).height;
        n = cvGetSize(A).width;
    }
    l = cvGetSize(B).width;

    
    if (tABC != CV_GEMM_A_T) {
      A_i_nonzero_indices = cached_row_nonzero_indices[A];
      if (A_i_nonzero_indices == 0) {
          //printf("A not found!\n");
          A_i_nonzero_indices = (vector<int> **)malloc(sizeof(vector<int> *) * m);
          for (i = 0; i < m; i++) {
              A_i_nonzero_indices[i] = new vector<int>;
              A_i_nonzero_indices[i]->reserve(5);              
              for (k = 0; k < n; k++) {
                  if (cvmGet(A,i,k) != 0) {
                      A_i_nonzero_indices[i]->push_back(k);
                  } 
              }
          }
          cached_row_nonzero_indices[A] = A_i_nonzero_indices;
      } else {
          //printf("A found!\n");
      }
    } else {
        A_i_nonzero_indices = cached_col_nonzero_indices[A];
        if (A_i_nonzero_indices == 0) {
            //printf("A transpose NOT found!\n");
            A_i_nonzero_indices = (vector<int> **)malloc(sizeof(vector<int> *) * m);
            for (i = 0; i < m; i++) {
                A_i_nonzero_indices[i] = new vector<int>;
                A_i_nonzero_indices[i]->reserve(5);              
                for (k = 0; k < n; k++) {
                    if (cvmGet(A,k,i) != 0) {
                        A_i_nonzero_indices[i]->push_back(k);
                    } 
                }
            }
            cached_col_nonzero_indices[A] = A_i_nonzero_indices;
        } else {
            //printf("A transpose found!\n");
        }
    }

    B_j_nonzero_indices = cached_col_nonzero_indices[B];
    if (B_j_nonzero_indices == 0) {
        //printf("B NOT found!\n");
        B_j_nonzero_indices = (vector<int> **)malloc(sizeof(vector<int> *) * l);
        for (j = 0; j < l; j++) {
            B_j_nonzero_indices[j] = new vector<int>;
            B_j_nonzero_indices[j]->reserve(5);
            for (k = 0; k < n; k++) {
                if (cvmGet(B,k,j) != 0) {
                    B_j_nonzero_indices[j]->push_back(k);
                }
            }
        }
        cached_col_nonzero_indices[B] = B_j_nonzero_indices;
    } else {
        //printf("B found!\n");
    }
    
    for (i = 0; i < m; i++ ) {
        for (j = 0; j < l; j++) {
            vector<int>::iterator A_i_index = A_i_nonzero_indices[i]->begin();
            vector<int>::iterator B_j_index = B_j_nonzero_indices[j]->begin();
            sum = 0;
            while (A_i_index != A_i_nonzero_indices[i]->end() &&
                   B_j_index != B_j_nonzero_indices[j]->end()) {
                if (*A_i_index == *B_j_index) {
                    if (tABC == CV_GEMM_A_T) {
                        sum += cvmGet(A,*A_i_index,i) * cvmGet(B,*B_j_index,j);
                    } else {
                        sum += cvmGet(A,i,*A_i_index) * cvmGet(B,*B_j_index,j);
                    }
                    A_i_index++;
                    B_j_index++;
                } else if (*A_i_index < *B_j_index) {
                    A_i_index++;
                } else {
                    B_j_index++;
                }
            }
            cvmSet(C,i,j,sum);
        }
    }
}

int SigmoidLogBarrierSolver(CvMat *vect_n1_w,
			    double *alpha, 
			    int *status,
			    CvMat *mat_mn_A,
			    CvMat *vect_m1_b,
			    CvMat *mat_ln_S,
			    CvMat *vect_l1_q,
			    double t_0 = 500,
			    double alpha_0 = 1e-1);

int main(int argc, char ** argv)
{
  int status, m, n, l, i;
  double alpha, *data;
  CvMat *A, *b, *S, *q, *w;
  MATFile *fh;
  mxArray *mat;

  cvUseOptimized(0);


  fh = matOpen("A.mat", "r");
  mat = matGetVariable(fh, "A");
  matClose(fh);
  A = createCvMatFromMatlab(mat);
  m = cvGetSize(A).height;
  n = cvGetSize(A).width;
  printMat("A", A);

  fh = matOpen("b.mat", "r");
  mat = matGetVariable(fh, "b");
  matClose(fh);
  b = createCvMatFromMatlab(mat);
  if ((n != cvGetSize(b).height) &&
      (cvGetSize(b).width != 1)) {
    printf("b and A must match dimensions\n");
    return -1;
  }
  printMat("b", b);
  

  fh = matOpen("S.mat", "r");
  mat = matGetVariable(fh, "S");
  matClose(fh);
  S = createCvMatFromMatlab(mat);
  l = cvGetSize(S).height;
  if (cvGetSize(S).width !=  n) {
    printf("Column size of S must match column size of A\n");
    return -1;
  }
  printMat("S", S);

  fh = matOpen("inq.mat", "r");
  mat = matGetVariable(fh, "inq");
  matClose(fh);
  q = createCvMatFromMatlab(mat);
  printMat("q", q);

  w = cvCreateMat(n, 1, CV_64FC1);

  SigmoidLogBarrierSolver(w, &alpha, &status, A, b, S, q);

  //  printMat("w", w);
}

/**
 * INPUT
 *
 *  mat_mn_A       : m x n matrix;
 *  vect_m1_b       : m vector; 
 *  mat_ln_S       : l x n matrix;
 *  vect_l1_q       : l vector;
 * OUTPUT
 *
 *  vect_w       : n vector; classifier
 * 
 *  minimize norm(Aw - b, 1) s.t., Sw+q<=0
 */
int SigmoidLogBarrierSolver(CvMat *vect_n1_w,
			    double *alpha, 
			    int *status,
			    CvMat *mat_mn_A, 
			    CvMat *vect_m1_b, 
			    CvMat *mat_ln_S, 
			    CvMat *vect_l1_q, 
			    double t_0, 
			    double alpha_0)
{
    int m, n, l, i, j, total_linesearch, ntiter, lsiter, totalIter, newtonLoop;
    double minElem, maxElem, alpha_max, lambdasqr, factor;

    m = cvGetSize(mat_mn_A).height;
    n = cvGetSize(mat_mn_A).width;
    l = cvGetSize(mat_ln_S).height;
    
    CvMat *vect_l1_tmp = cvCreateMat(l, 1, CV_64FC1);
    CvMat *vect_1l_tmp = cvCreateMat(1, l, CV_64FC1);
    CvMat *vect_m1_tmp = cvCreateMat(m, 1, CV_64FC1);
    CvMat *vect_1m_tmp = cvCreateMat(1, m, CV_64FC1);
    CvMat *vect_1n_tmp = cvCreateMat(1, n, CV_64FC1);
    CvMat *vect_n1_tmp = cvCreateMat(n, 1, CV_64FC1);
    CvMat *vect_n1_new_w = cvCreateMat(n, 1, CV_64FC1);
    CvMat *vect_n1_dw = cvCreateMat(n, 1, CV_64FC1);
    CvMat *vect_l1_q_tilde = cvCreateMat(l, 1, CV_64FC1);
    CvMat *vect_1m_logExpTerm = cvCreateMat(1, m, CV_64FC1);
    CvMat *vect_1m_expTerm = cvCreateMat(1, m, CV_64FC1);
    CvMat *vect_1m_expTerm_Rec = cvCreateMat(1, m, CV_64FC1);
    CvMat *vect_1m_expTermNeg_Rec = cvCreateMat(1, m, CV_64FC1);
    CvMat *vect_n1_gradphi_sigmoid = cvCreateMat(n, 1, CV_64FC1);
    CvMat *vect_n1_gradphi = cvCreateMat(n, 1, CV_64FC1);
    CvMat *vect_1l_g_t = cvCreateMat(1, l, CV_64FC1);
    CvMat *vect_1l_h_t = cvCreateMat(1, l, CV_64FC1);
    CvMat *vect_1m_h_sigmoid = cvCreateMat(1, m, CV_64FC1);
    CvMat *mat_mn_tmp = cvCreateMat(m, n, CV_64FC1);
    CvMat *mat_nn_tmp = cvCreateMat(n, n, CV_64FC1);
    CvMat *mat_nn_hessphi = cvCreateMat(n, n, CV_64FC1);
    CvMat *mat_ln_tmp = cvCreateMat(l, n, CV_64FC1);
    
    //------------------------------------------------------------
    //        INITIALIZATION
    //------------------------------------------------------------
    
    // LOG BARRIER METHOD
    double EPSILON_GAP = 5e-5;
    double MU_t = 3;
    double t = 500;

    // SIGMOID APPROXIMATION
    double NUMERICAL_LIMIT_EXP = 3e2;
    double MU_alpha = 1;
    double MU_alpha2 = 2;
    double ALPHA_MAX = 3000;
    double ALPHA_MIN = 500;
    
    // NEWTON PARAMETERS
    double MAX_TNT_ITER = 50; 
    double EPSILON = 1e-6;
    
    // LINE SEARCH PARAMETERS
    double ALPHA_LineSearch = 0.01;
    double BETA = 0.5;
    double MAX_LS_ITER = 25;
    double s0 = 1;
    double s;
    
    // VARIABLE INITIALIZE
    for (i = 0; i < n; i++) {
        switch (i % 3) {
        case 0 :
            cvmSet(vect_n1_w, i, 0, 0);
            break;
        case 1:
            cvmSet(vect_n1_w, i, 0, -0.1);
            break;
        case 2:
            cvmSet(vect_n1_w, i, 0, 1);
            break;
        }
    }
    
    // ll = S*w;
    // factor = max(ll( Para.Dist_Start:end));
    // factor = 0.9/factor;
    cvMatMul(mat_ln_S, vect_n1_w, vect_l1_tmp);

    factor = cvmGet(vect_l1_tmp, (n/3)+1, 0);
    for (i = (n/3)+1; i < l; i++) {
        if (cvmGet(vect_l1_tmp, i, 0) > factor)
            factor = cvmGet(vect_l1_tmp, i, 0);
    }
    factor = 0.9/factor;
    cvConvertScale(vect_n1_w, vect_n1_w, factor);

    //q_tilde = q + 1e-15;
    cvAddS(vect_l1_q, cvScalar(1e-15), vect_l1_q_tilde);
    
    //if max( S*w+q_tilde) >= 0...
    cvMatMul(mat_ln_S, vect_n1_w, vect_l1_tmp);
    cvAdd(vect_l1_tmp, vect_l1_q_tilde, vect_l1_tmp);
    cvMinMaxLoc(vect_l1_tmp, &minElem, &maxElem);
    if (maxElem > 0) {
        printf("Unable to initialize variables.  Returning!\n");
        goto EXIT_LABEL;
    }
    
    //Setting starting alpha_0
    //alpha_0 = 1 / max( abs(A*w-b) );
    cvMatMul( mat_mn_A, vect_n1_w, vect_m1_tmp);
    cvAbsDiff(vect_m1_tmp, vect_m1_b, vect_m1_tmp);
    cvMinMaxLoc(vect_m1_tmp, &minElem, &maxElem);

    alpha_0 = 1/maxElem;
    
    //alfa = alpha_0;
    *alpha = alpha_0;

    //alfa_max = alfa;
    alpha_max = *alpha;
    
    *status = -1;
    totalIter = 0;
    while (*status != 2) {
        //dw =  zeros(n,1); % dw newton step
        totalIter++;
        printf("totalIter: %d\n", totalIter);

        cvSetZero(vect_n1_dw);
        
        total_linesearch = 0;
        ntiter = 0;

        //logExpTerm = alfa*(A*w-b)';
        cvGEMM(mat_mn_A, vect_n1_w, *alpha, vect_m1_b, -1*(*alpha), vect_m1_tmp);
        cvTranspose(vect_m1_tmp, vect_1m_logExpTerm);


        //expTerm = exp(logExpTerm);
        cvExp(vect_1m_logExpTerm, vect_1m_expTerm);
        
        //expTerm_Rec = 1./(1+expTerm);
        cvAddS(vect_1m_expTerm, cvScalar(1), vect_1m_tmp);

        //cvDiv(0, vect_1m_tmp, vect_1m_expTerm_Rec, 1);
        for (i = 0; i < m; i++) {
            cvmSet(vect_1m_expTerm_Rec, 0, i, 1/cvmGet(vect_1m_tmp, 0, i));
        }
        
        //expTermNeg_Rec = 1./(1+exp(-logExpTerm));
        cvConvertScale(vect_1m_logExpTerm, vect_1m_tmp, -1);
        cvExp(vect_1m_tmp, vect_1m_tmp);
        cvAddS(vect_1m_tmp, cvScalar(1), vect_1m_tmp);
        //cvDiv(0, vect_1m_tmp, vect_1m_expTermNeg_Rec, 1);
        for (i = 0; i < m; i++) {
            cvmSet(vect_1m_expTermNeg_Rec, 0, i, 1/cvmGet(vect_1m_tmp, 0, i));
        }

        
        //g_sigmoid = (expTermNeg_Rec - expTerm_Rec) * t;
        //gradphi_sigmoid = (g_sigmoid*A)'; %A1endPt) + ...
        cvSub(vect_1m_expTermNeg_Rec, vect_1m_expTerm_Rec, vect_1m_tmp);
        cvConvertScale(vect_1m_tmp, vect_1m_tmp, t);
        cvMatMul(vect_1m_tmp, mat_mn_A, vect_1n_tmp);
        cvTranspose(vect_1n_tmp, vect_n1_gradphi_sigmoid);
        
        //inequalityTerm = (S*w+q)';
        //g_t = (-1./inequalityTerm);       % log barrier  
        //gradphi_t = (g_t*S)';
        cvMatMul(mat_ln_S, vect_n1_w, vect_l1_tmp);
        cvAdd(vect_l1_tmp, vect_l1_q, vect_l1_tmp);
        cvTranspose(vect_l1_tmp, vect_1l_tmp);
        //cvDiv(0, vect_1l_tmp, vect_1l_g_t, -1);
        for (i = 0; i < l; i++) {
            cvmSet(vect_1l_g_t, 0, i, -1/cvmGet(vect_1l_tmp, 0, i));
        }
        
        cvMatMul(vect_1l_g_t, mat_ln_S, vect_1n_tmp);
        cvTranspose(vect_1n_tmp, vect_n1_gradphi);
        cvAdd(vect_n1_gradphi, vect_n1_gradphi_sigmoid, vect_n1_gradphi);

        newtonLoop = (ntiter <= MAX_TNT_ITER)?1:0;
        while (newtonLoop) {
            ntiter++;

            printf("\tnewton loop iter: %d\n", ntiter);
            
            for (i = 0; i < m; i++) {
                if ((cvmGet(vect_1m_logExpTerm, 0, i) > NUMERICAL_LIMIT_EXP) ||
                    (cvmGet(vect_1m_logExpTerm, 0, i) < (-1)*NUMERICAL_LIMIT_EXP)) {
                    cvmSet(vect_1m_h_sigmoid, 0, i, 0);
                } else {
                    double expTerm_Rec_i = cvmGet(vect_1m_expTerm_Rec, 0, i);
                    double expTerm_i = cvmGet(vect_1m_expTerm, 0, i);
                    cvmSet(vect_1m_h_sigmoid, 0, i,
                           expTerm_Rec_i*expTerm_Rec_i*expTerm_i*t);
                }
            }
            
            //hessphi_sigmoid =  (sparse(1:m,1:m,h_sigmoid)*A)' * A;
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    cvmSet(mat_mn_tmp, i, j, 
                           cvmGet(vect_1m_h_sigmoid, 0, i)*cvmGet(mat_mn_A, i, j));
                }
            }
            if (DEBUG_PERF_MON) printf("Calling cvSparseMatMul to calculate hessphi_sigmoid\n");
            cvSparseMatMul(mat_mn_tmp, mat_mn_A, mat_nn_tmp, CV_GEMM_A_T);
            if (DEBUG_PERF_MON) printf("return from cvSparseMatMul to calculate hessphi_sigmoid\n");
            
            //h_t = g_t.^2;% log barrier
            cvPow(vect_1l_g_t, vect_1l_h_t, 2);
            
            if (DEBUG_PERF_MON) printf("return from cvPow\n");
            
            //hessphi_t = ( (sparse(1:length(h_t), 1:length(h_t), h_t) * S )' * S ) / (2*alfa);
            vector<int> **S_nonzero_col_indices_in_row = 0;
            S_nonzero_col_indices_in_row = cached_row_nonzero_indices[mat_ln_S];
            if (S_nonzero_col_indices_in_row == 0) {
                if (DEBUG) printf("S not found in cache.\n");
                S_nonzero_col_indices_in_row = (vector<int> **)malloc(sizeof(vector<int> *) * l);
                for (i = 0; i < l; i++) {
                    S_nonzero_col_indices_in_row[i] = new vector<int>;
                    S_nonzero_col_indices_in_row[i]->reserve(5);                
                    for (j = 0; j < n; j++) {
                        if (cvmGet(mat_ln_S,i,j) != 0) {
                            S_nonzero_col_indices_in_row[i]->push_back(j);
                        } 
                    }
                }
                cached_row_nonzero_indices[mat_ln_S] = S_nonzero_col_indices_in_row;
            } else {
                if (DEBUG) printf("S found in cache!.\n");
            }
            
            cvSetZero(mat_ln_tmp);
            if (DEBUG_PERF_MON) printf("after setting mat_ln_tmp to zero!.\n");
            
            for (i = 0; i < l; i++) {
                vector<int>::iterator S_nonzero_col_index = S_nonzero_col_indices_in_row[i]->begin();
                if (cvmGet(vect_1l_h_t, 0, i) != 0) {
                    while (S_nonzero_col_index != S_nonzero_col_indices_in_row[i]->end()) {
                        cvmSet(mat_ln_tmp, i, *S_nonzero_col_index, cvmGet(vect_1l_h_t, 0, i)*cvmGet(mat_ln_S, i, *S_nonzero_col_index));
                        S_nonzero_col_index++;
                    }
                }
            }
            
            if (DEBUG_PERF_MON) printf("Calling cvSparseMatMul to calculate hessphi_t\n");
            cvSparseMatMul(mat_ln_tmp, mat_ln_S, mat_nn_hessphi, CV_GEMM_A_T);
            if (DEBUG_PERF_MON) printf("return from cvSparseMatMul to calculate hessphi_t\n");
            
            cvAddWeighted(mat_nn_hessphi, 1/(2*(*alpha)), mat_nn_tmp, 1, 0, mat_nn_hessphi);
            
            if (DEBUG_PERF_MON) printf("Done calculating hessphi_t!\n");
            
            cvInvert(mat_nn_hessphi, mat_nn_tmp);
            if (DEBUG_PERF_MON) printf("Done Inverting!\n");
            
            cvGEMM(mat_nn_tmp, vect_n1_gradphi, (-1)/(2*(*alpha)), 0, 0, vect_n1_dw);
            
            lambdasqr = cvDotProduct(vect_n1_gradphi, vect_n1_dw)*-1;

            //------------------------------------------------------------
            //   BACKTRACKING LINE SEARCH
            //------------------------------------------------------------
            s = s0;
            bool someGreaterThan0;
            do {
                cvAddWeighted(vect_n1_w, 1, vect_n1_dw, s, 0, vect_n1_tmp);
                cvMatMul(mat_ln_S, vect_n1_tmp, vect_l1_tmp);
                cvAdd(vect_l1_tmp, vect_l1_q_tilde, vect_l1_tmp);
                someGreaterThan0 = false;
                for (i = 0; i < l; i++) {
                    if (cvmGet(vect_l1_tmp, i, 0) >= 0) {
                        someGreaterThan0 = true;
                        break;
                    }
                }
                if (someGreaterThan0) s = BETA * s;
            } while (someGreaterThan0);

            double norm_gradphi = cvNorm(vect_n1_gradphi);
            lsiter = 0;
            bool backIterationLoop = true;
            while (backIterationLoop) {
                lsiter++;
                
                printf("\t\tback line search iter: %d\n", lsiter);
        
                cvAddWeighted(vect_n1_w, 1, vect_n1_dw, s, 0, vect_n1_new_w);
          
                //logExpTerm = alfa*(A*w-b)';
                cvGEMM(mat_mn_A, vect_n1_new_w, *alpha, vect_m1_b, -1*(*alpha), vect_m1_tmp);
                cvTranspose(vect_m1_tmp, vect_1m_logExpTerm);
                
                //expTerm = exp(logExpTerm);
                cvExp(vect_1m_logExpTerm, vect_1m_expTerm);
                
                //expTerm_Rec = 1./(1+expTerm);
                cvAddS(vect_1m_expTerm, cvScalar(1), vect_1m_tmp);
		//                cvDiv(0, vect_1m_tmp, vect_1m_expTerm_Rec, 1);
                for (i = 0; i < m; i++) {
                    cvmSet(vect_1m_expTerm_Rec, 0, i, 1/cvmGet(vect_1m_tmp, 0, i));
                }
                
                //expTermNeg_Rec = 1./(1+exp(-logExpTerm));
                cvConvertScale(vect_1m_logExpTerm, vect_1m_tmp, -1);
                cvExp(vect_1m_tmp, vect_1m_tmp);
                cvAddS(vect_1m_tmp, cvScalar(1), vect_1m_tmp);
		//                cvDiv(0, vect_1m_tmp, vect_1m_expTermNeg_Rec, 1);
                for (i = 0; i < m; i++) {
                    cvmSet(vect_1m_expTermNeg_Rec, 0, i, 1/cvmGet(vect_1m_tmp, 0, i));
                }

                
                //g_sigmoid = (expTermNeg_Rec - expTerm_Rec) * t;
                //gradphi_sigmoid = (g_sigmoid*A)'; %A1endPt) + ...
                cvSub(vect_1m_expTermNeg_Rec, vect_1m_expTerm_Rec, vect_1m_tmp);
                cvConvertScale(vect_1m_tmp, vect_1m_tmp, t);
                cvMatMul(vect_1m_tmp, mat_mn_A, vect_1n_tmp);
                cvTranspose(vect_1n_tmp, vect_n1_gradphi_sigmoid);
                
                //inequalityTerm = (S*w+q)';
                //g_t = (-1./inequalityTerm);       % log barrier  
                //gradphi_t = (g_t*S)';
                cvMatMul(mat_ln_S, vect_n1_new_w, vect_l1_tmp);
                cvAdd(vect_l1_tmp, vect_l1_q, vect_l1_tmp);
                cvTranspose(vect_l1_tmp, vect_1l_tmp);
                //cvDiv(0, vect_1l_tmp, vect_1l_g_t, -1);
                for (i = 0; i < l; i++) {
                    cvmSet(vect_1l_g_t, 0, i, -1/cvmGet(vect_1l_tmp, 0, i));
                }
                
                cvMatMul(vect_1l_g_t, mat_ln_S, vect_1n_tmp);
                cvTranspose(vect_1n_tmp, vect_n1_gradphi);
                cvAdd(vect_n1_gradphi, vect_n1_gradphi_sigmoid, vect_n1_gradphi);
                
                backIterationLoop = (lsiter <= MAX_LS_ITER 
                                     && cvNorm(vect_n1_gradphi) > (1-ALPHA_LineSearch*s)*norm_gradphi);
                s = BETA * s;
            }
            
            total_linesearch += lsiter;
            if (lambdasqr/2 <= EPSILON) {
                *status = 1;
            }
            newtonLoop = ((ntiter <= MAX_TNT_ITER) &&
                          (lambdasqr/2 > EPSILON) &&
                          (lsiter < MAX_LS_ITER));

            cvCopy(vect_n1_new_w, vect_n1_w);
        }

        double gap = m / t;

        if ((*alpha > ALPHA_MIN) && (gap < EPSILON_GAP) && (*status >=1)) {
            *status = 2;
        }
        t = MU_t * t;
        *alpha = ((*alpha*MU_alpha2) < ALPHA_MAX)?(*alpha*MU_alpha2):ALPHA_MAX;
        alpha_max = ((*alpha*MU_alpha2) < ALPHA_MAX)?(*alpha*MU_alpha2):ALPHA_MAX;
        printf("s: %f alpha: %f alpha_max: %f t: %f\n", s, alpha, alpha_max, t);
    }
    if (DEBUG) {
      printf("final w:\n");
    }

 EXIT_LABEL:
    cvReleaseMat(&vect_1l_tmp);
    cvReleaseMat(&vect_m1_tmp);
    cvReleaseMat(&vect_1m_tmp);
    cvReleaseMat(&vect_1n_tmp);
    cvReleaseMat(&vect_n1_tmp);
    cvReleaseMat(&vect_n1_new_w);
    cvReleaseMat(&vect_n1_dw);
    cvReleaseMat(&vect_l1_q_tilde);
    cvReleaseMat(&vect_1m_logExpTerm);
    cvReleaseMat(&vect_1m_expTerm);
    cvReleaseMat(&vect_1m_expTerm_Rec);
    cvReleaseMat(&vect_1m_expTermNeg_Rec);
    cvReleaseMat(&vect_n1_gradphi_sigmoid);
    cvReleaseMat(&vect_n1_gradphi);
    cvReleaseMat(&vect_1l_g_t);
    cvReleaseMat(&vect_1l_h_t);
    cvReleaseMat(&vect_1m_h_sigmoid);
    cvReleaseMat(&mat_mn_tmp);
    cvReleaseMat(&mat_nn_tmp);
    cvReleaseMat(&mat_nn_hessphi);
    cvReleaseMat(&mat_ln_tmp);
    return 0;
}

void printMat(char*name, CvMat * mat)
{
  if (DEBUG_MATRIX) {
    printf("%s %d x %d:\n", name, cvGetSize(mat).height, cvGetSize(mat).width);
    for (int i = 0; i < cvGetSize(mat).height; i++) {
      for (int j = 0; j < cvGetSize(mat).width; j++) {
	printf("%f ", cvmGet(mat, i, j));
      }
      printf("\n");
    }
  }
}

CvMat* createCvMatFromMatlab(const mxArray *mxArray)
{
  double  *pr, *pi;
  mwIndex  *ir, *jc;
  mwSize      col, total=0;
  mwIndex   starting_row_index, stopping_row_index, current_row_index;
  mwSize      m, n;
  CvMat *mat;

  m = mxGetM(mxArray);
  n = mxGetN(mxArray);

  mat = cvCreateMat(m, n, CV_64FC1);
  cvSetZero(mat);

  /* Get the starting positions of all four data arrays. */
  pr = mxGetPr(mxArray);
  pi = mxGetPi(mxArray);
  ir = mxGetIr(mxArray);
  jc = mxGetJc(mxArray);

  /* Display the nonzero elements of the sparse array. */
  for (col=0; col<n; col++) {
    starting_row_index = jc[col];
    stopping_row_index = jc[col+1];
    if (starting_row_index == stopping_row_index)
      continue;
    else {
      for (current_row_index = starting_row_index;
           current_row_index < stopping_row_index;
           current_row_index++) {
	  cvmSet(mat, ir[current_row_index], col, pr[total]);
          if (DEBUG) {
	    printf("\t(%"FMT_SIZE_T"u,%"FMT_SIZE_T"u) = %g\n", ir[current_row_index]+1,
		   col+1, pr[total]);
	  }
	  total++;
      }
    }
  }
  return mat;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  int status, m, n, l, i;
  double alpha, *data;
  CvMat *A, *b, *S, *q, *w;

  if (nrhs != 4) {
    printf("Invalid arguments!\n");
    return;
  }

  //A
  A = createCvMatFromMatlab(MAT_A);
  m = cvGetSize(A).height;
  n = cvGetSize(A).width;
  printMat("A", A);
  
  //b
  b = createCvMatFromMatlab(VECT_B);
  if ((n != cvGetSize(b).height) &&
      (cvGetSize(b).width != 1)) {
    printf("b and A must match dimension\n");
    return;
  }
  printMat("b", b);

  //S
  S = createCvMatFromMatlab(MAT_S);
  l = cvGetSize(S).height;
  if (cvGetSize(S).width !=  n) {
    printf("Column size of S must match column size of A\n");
    return;
  }
  printMat("S", S);

  //q
  q = createCvMatFromMatlab(VECT_Q);
  if ((l != cvGetSize(q).height) &&
      (cvGetSize(q).width != 1)) {
    printf("b and A must match dimension\n");
    return;
  }

  printMat("q", q);


  //w
  w = cvCreateMat(n, 1, CV_64FC1);

  SigmoidLogBarrierSolver(w, &alpha, &status, A, b, S, q);

  VECT_W = mxCreateDoubleMatrix(n, 1, mxREAL);
  data = mxGetPr(VECT_W);
  for (i = 0; i < n; i++) {
    data[i] = cvmGet(w, i, 0);
  }

  DOUBLE_ALPHA = mxCreateDoubleMatrix(1, 1, mxREAL);
  mxGetPr(DOUBLE_ALPHA)[0] = alpha;

  INT_STATUS = mxCreateDoubleMatrix(1, 1, mxREAL);
  mxGetPr(INT_STATUS)[0] = status;

  cvReleaseMat(&A);
  cvReleaseMat(&b);
  cvReleaseMat(&S);
  cvReleaseMat(&q);
  cvReleaseMat(&w);
}
