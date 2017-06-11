function example2_noR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      minimize S(x)   subject to Wx = x   
%    S is differentiable: S = 1/2||Mx-y||_2^2             
%    W is the given mixing matrix             
       
%    Reference: A Decentralized Proximal-Gradient Method with Network 
%               Independent Step-zsizes and Seperated Convergence Rates
%       
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global n m p M y_ori lam
path(path, '.\fcns')

n = 40; % number of nodes
m = 60;
p = 50; % the dimension of x on each nodes

L = n;
per = 16/L;
resSubPath = 'per16overL_mu0_2';
% call which function to change M
modifyM_num = 7;
Mrate = 2;
% may changed in the following function
min_mu = 0.2; % set the smallest strongly convex parameter mu in S
max_Lips = 1; % set the Lipschitz constant

W = generateW(L,per);
[M, x_ori, y_ori] = generateS(m, p, n,...
    'withoutNonsmoothR',min_mu,max_Lips,Mrate,modifyM_num);

[~, lambdan] = eigW(W); % find the smallest eigenvalue of W

[Lips,mus] = getBetaSmoothAlphaStrong;
max_Lips   = max(Lips);
min_mu     = min(mus);

% set parameters
iter    = 200;
tol     = 1e-11;     % tolerance, this controls |x-x_star|_F, not divided by |x_star|_F
x0      = zeros(n,p);
x_star  = x_ori;     % true solution
% Set the parameter for the solver
paras.min_mu    = min_mu;
paras.max_Lips  = max_Lips;
paras.x_star    = x_star;
paras.n         = n;    % the number of nodes
paras.p         = p;    % the dimension of x on each nodes
paras.iter      = iter; % max iteration
paras.x0        = x0;   % the initial x
paras.W         = W;    % the mixing matrix
paras.tol       = tol;  % tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start using the NIDS class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
obj            =  NIDS;  % using the class PrimalDual

obj.getS       = @(x) feval(@funS, x);
obj.getGradS   = @(x) feval(@funGradS, x);

h   = figure;
set(h, 'DefaultLineLineWidth', 4)
norm_x_star = norm(x_star, 'fro');

methods = {'NIDSS-adaptive','NIDSS','EXTRA','DIGing-ATC'};
LineSpecs = {'-k','--k','-.b',':m','-.y'};

numMethods = length(methods);
outputs = cell(numMethods,1);
legend_lab = cell(numMethods,1);
% call methods
for i = 1:numMethods
    paras.method = methods{i};
    
    switch methods{i}
        case {'DIGing-ATC'}
            paras.alpha = cRate./max_Lips*ones(n,1);
            paras.atc = 1;
            outputs{i}  = obj.minimize_DIGing(paras);
            legend_lab{i} = ['DIGing-ATC-',num2str(cRate),'/L'];
        
        case {'DIGing'}
            paras.alpha = cRate./max_Lips*ones(n,1);
            paras.atc = 0;
            outputs{i}  = obj.minimize_DIGing(paras);
            
        case {'NIDSS-adaptive'}
            cRate = 1; % rate of the step size < 2
            alpha = cRate./Lips;
            paras.alpha = alpha;
            
            eye_L = eye(n);
            I_W   = eye_L-W;
            [U,S,V] = svd(I_W);
            a = diag(S);
            inv_I_W = U*diag([a(1:end-1).^(-1);0])*V';
            alpha2 = diag(sqrt(1./alpha));
            eigs = eig(alpha2*inv_I_W*alpha2);
            [~, indmin]=min(eigs);
            ind = ones(size(eigs)); ind(indmin) = 0;
            lambda_n_1 = min(eigs(logical(ind)));
            c= lambda_n_1;
            
            paras.c     = c;
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = ['NIDS-adaptive'];
            
        case {'NIDSS'}
            cRate = 1; % rate of the step size < 2
            
            alpha = cRate./max_Lips*ones(n,1);
            c = 1/(1-lambdan)/max(alpha);
            paras.alpha = alpha;
            paras.c = c;
            
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = ['NIDS-',num2str(cRate),'/L'];
        case {'NIDSS-F'}
            cRate = 1; % rate of the step size < 2
            
            alpha = cRate./max_Lips*ones(n,1);
            paras.alpha = alpha;
            
            paras.forcetTildeW = true;
            paras.method = 'NIDSS';
            
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = ['NIDS-$\widetilde W = {I+W\over 2}$-',num2str(cRate),'/L'];
            
        case {'EXTRA'}
            cRate = 1;
            
            paras.alpha = cRate./max_Lips*ones(n,1);
            outputs{i} = obj.minimize(paras);
            legend_lab{i} = ['EXTRA-',num2str(cRate),'/L'];
        otherwise
            display('????')
    end
end
for i = 1:numMethods
    semilogy(outputs{i}.err/norm_x_star,LineSpecs{i});
    hold on;
end

xlabel('number of iterations');
ylabel('$\frac{\left\Vert \mathbf{x}-\mathbf{x}^{*}\right\Vert}{\left\Vert \mathbf{x}^{*}\right\Vert}$','FontSize',20,'Interpreter','LaTex');

legend(legend_lab,'FontSize',10,'Interpreter','LaTex');

saveas(h,[resSubPath,'_compa3.fig']);
% close;
prob.M = M;
prob.x_ori = x_ori;
prob.y_ori = y_ori;
prob.lam = lam;
prob.W = W;

save([resSubPath,'_compa3_prob.mat'],'prob');
end

function a = funGradS(x)
global n p M y_ori
a = zeros(n, p);
for j = 1:n
    a(j,:) = (M(:,:,j)' * (M(:,:,j) * (x(j,:))' - y_ori(:,j)))';
end
end

function a = funS(x)
global n M y_ori
a = 0;
for j = 1:n
    a   = a + 0.5 * sum((M(:,:,j) * (x(j,:))' - y_ori(:,j)).^2);
end
end
