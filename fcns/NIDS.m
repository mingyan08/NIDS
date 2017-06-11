classdef NIDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Decentralized Optimization with Network InDependent Stepsizes                             
%      minimize S(x) + R(x)   subject to Wx = x                                              
%    S is differentiable and R is proximable, W is the given mixing matrix             
%
%
%    Three methods are implemented: PG-EXTRA, NIDS, and DIGing         
%    Reference: 
%       
%
%    Contact:
%       Zhi Li lizhi@msu.edu
%       Wei Shi ??? 
%       Ming Yan myan@math.msu.edu
% 
%       Downloadable from ?????                                                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        getGradS   =  @(x) 0;            % gradient of S      gradS: x  --> grad(S)(x)
        getProxR   =  @(x,t) x;          % prox of R          proxR: x,t  --> prox(t.R)(x)
        getS       =  @(x) 0;            % smooth function    S: x --> S(x)
        getR       =  @(x) 0;            % proximable fcn     R: x --> R(x)
    end
    
    methods
        function e = E(this, x)    % return the objective function value
            e   = this.getS(x) + this.getR(x);                       
        end
        
        function [out] = minimize(this, paras)
            % get all the parameters
            tol     = paras.tol;    % tolerance
            n       = paras.n;      % the number of nodes
            p       = paras.p;      % the dimension of x on each nodes
            x_star  = paras.x_star; % the optimal solution x
            iter    = paras.iter;   % the maximum number of iterations
            x0      = paras.x0;     % the initial x
            alpha   = paras.alpha;  % the stepsize, which can be different for different nodes
            method  = paras.method; % the method that used 
            W       = paras.W;      % the mixing matrix
            
            Q       = diag(1./alpha);
            eye_L   = eye(n);
            I_W     = eye_L - W;
            
            if isfield(paras, 'forcetTildeW') && paras.forcetTildeW
                tildeW = (eye_L + W)/2;
            else
                switch method
                    case {'NIDS', 'NIDS-adaptive', 'NIDSS', 'NIDSS-adaptive'}  % NIDS
                        c = paras.c;      % the parameter c in  NIDS

                        tildeW = eye_L - c * diag(alpha) * I_W;
                    case {'PG-EXTRA', 'PGEXTRA', 'EXTRA'}
                        tildeW = (eye_L + W)/2;
                end
            end
            
            x_cur   = x0;           % the current iterate   x^k 
            gradS   = this.getGradS(x_cur); % the gradient of the current iterate

            err     = zeros(iter + 2,1);   % store the L2 norm with x_star
            eng     = zeros(iter + 2,1);   % store the objective function values            
            err(1)  = norm(x_cur - x_star,'fro'); 
            eng(1)  = this.E(x_cur);  
            x_array = cell(iter,1); % store all x values
            d_array = cell(iter,1); % intermediate variable for showing the rate of convergence
            
            convergeTol = 0;   % reach convergence tolerance or not

            % initialization for the 1st step,
            switch method
                case {'NIDS', 'NIDS-adaptive', 'NIDSS', 'NIDSS-adaptive'} 
                    z       = x_cur - diag(alpha) * gradS;  % 
                    x_new   = this.getProxR(z, alpha);      % 
                    x_old   = x_cur;                        % increase k by 1
                    x_cur   = x_new;                        % increase k by 1 
                    gradS_old = gradS;                      % increase k by 1
                    gradS   = this.getGradS(x_cur);         % increase k by 1
                case {'PG-EXTRA', 'PGEXTRA', 'EXTRA'}
                    z       = W * x_cur - diag(alpha) * gradS; 
                    x_new   = this.getProxR(z, alpha);     
                    x_old   = x_cur;                        % increase k by 1
                    x_cur   = x_new;                        % increase k by 1 
                    gradS_old = gradS;                      % increase k by 1
                    gradS   = this.getGradS(x_cur);         % increase k by 1
                otherwise
                    warning('Unexpected Method, Please choose from ?? ')                    
            end
            err(2)  = norm(x_cur - x_star,'fro');
            eng(2)  = this.E(x_cur);  
         
            for i=1:iter
                % x_new x^{k+1}, x_cur x^{k}, x_old x^{k-1}
                %                gradS {k}, gradS_old {k-1}
                switch method
                    case {'NIDS', 'NIDS-adaptive', 'NIDSS', 'NIDSS-adaptive'}  
                        x_bar = 2 * x_cur - x_old - diag(alpha) * (gradS - gradS_old);
                        z     = z - x_cur + tildeW * x_bar;
                        x_new = this.getProxR(z, alpha);

                    case {'PG-EXTRA', 'PGEXTRA', 'EXTRA'}
                        x_bar = 2 * x_cur - x_old; 
                        z = z - x_cur + tildeW * x_bar - diag(alpha) * (gradS - gradS_old);
                        x_new = this.getProxR(z, alpha);                        
                    otherwise
                        warning('Unexpected Method, Please choose from ?? ')
                end
                
                x_array{i}  = x_new;                
                d_array{i}  = Q * (x_cur - x_new) - gradS;  % compute d based on (3.2) in the paper
                x_old       = x_cur;                    % increase k by 1    
                x_cur       = x_new;                    % increase k by 1
                gradS_old   = gradS;                    % increase k by 1
                gradS       = this.getGradS(x_cur);     % increase k by 1
                err(i + 2)  = norm(x_cur-x_star,'fro');
                eng(i + 2)  = this.E(x_cur); 
                
                if err(i + 2) < tol
                    x_array = x_array(1:i);
                    d_array = d_array(1:i);                    
                    err 	= err(1:(i + 2));
                    eng 	= eng(1:(i + 2));
                    convergeTol = 1;
                    break
                end
            end;
            out.err 	= err;
            out.eng 	= eng;
            out.x_array = x_array;
            out.d_array = d_array;
            out.convergeTol = convergeTol;
        end

        function [out] = minimize_DIGing(this, paras)
            tol     = paras.tol;    % tolerance
            x_star  = paras.x_star; % the optimal solution x
            x_cur   = paras.x0;     % the initial x
            gradS   = this.getGradS(x_cur); % the gradient of the current iterate
            y_cur   = gradS;
            alpha   = paras.alpha;  % the stepsize, which can be different for different nodes
            iter    = paras.iter;   % the maximum number of iterations            
            W       = paras.W;      % the mixing matrix            
            atc     = paras.atc;    % whether to choose ATC or not
            err     = zeros(iter + 1,1);   % store the L2 norm with x_star
            eng     = zeros(iter + 1,1);   % store the objective function values            
            err(1)  = norm(x_cur - x_star,'fro'); 
            eng(1)  = this.E(x_cur);  
            x_array = cell(iter,1); % store all x values
            
            convergeTol = 0;                    % reach convergence tolerance or not

            for i=1:iter
                if atc == 1
                    x_new = W * (x_cur - diag(alpha) * y_cur);
                    gradS_new = this.getGradS(x_new);
                    y_new = W * (y_cur + gradS_new - gradS);
                else
                    x_new = W * x_cur - diag(alpha) * y_cur;
                    gradS_new = this.getGradS(x_new);
                    y_new = W * y_cur + gradS_new - gradS;
                end
                x_array{i} = x_new;
                
                x_cur   = x_new;                          % increase k by 1
                y_cur   = y_new;                          % increase k by 1
                gradS   = gradS_new;                      % increase k by 1
                err(i+1) = norm(x_cur-x_star,'fro');
                eng(i+1) = this.E(x_cur);  
                
                if err(i+1)<tol
                    x_array = x_array(1:i);
                    err = err(1:(i+1));
                    eng = eng(1:(i+1));
                    convergeTol = 1;
                    break
                end
            end                        
            out.err = err;
            out.eng = eng;
            out.x_array = x_array;
            out.convergeTol = convergeTol;
        end
       
    end
end
