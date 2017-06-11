function example3_withR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      minimize S(x) + R(x)   subject to Wx = x                                              
%    S is differentiable: S = 1/2||Mx-y||_2^2 
%    R is proximable:     R = lam* ||x||_1
%    W is the given mixing matrix      
       
%    Reference: A Decentralized Proximal-Gradient Method with Network 
%               Independent Step-zsizes and Seperated Convergence Rates
%       
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global n m p M y_ori lam
path(path, '.\fcns')

n = 40;
m = 3;
p = 200;

L = n;
per = 4/L;
resSubPath = 'per4overL_mu0_1';

% may changed in the following function
min_mu = 0.1; % set the smallest strongly convex parameter mu in S
max_Lips = 1; % set the Lipschitz constant

% lam is the parameter in function R
% [M, x_ori, y_ori, lam, W] = generateAll(m, p, n, per,...
%     'withNonsmoothR', min_mu,max_Lips);
W = generateW(L,per);
[M, x_ori, y_ori, lam] = generateS(m, p, n,...
    'withNonsmoothR',min_mu,max_Lips);

[~, lambdan] = eigW(W); % find the smallest eigenvalue of W

% find the max Lipschitz constants and strongly convex parameters of the function S
[Lips,mus] = getBetaSmoothAlphaStrong;
max_Lips = max(Lips);
min_mu = min(mus);
% set parameters
iter    = 1000;
tol     = 1e-7;     % tolerance, this controls |x-x_star|_F, not divided by |x_star|_F

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
obj.getR       = @(x) feval(@funR, x);
obj.getGradS   = @(x) feval(@funGradS, x);
obj.getProxR   = @(x,t) feval(@funProxR, x, t);

h   = figure;
set(h, 'DefaultLineLineWidth', 4)
norm_x_star = norm(x_star, 'fro');

methods = {'NIDS','PGEXTRA'};
numMethods = length(methods);


cRate_NIDS = [1, 1.5, 1.9];
LineSpecs_NIDS = {':k','--k','-k','-*r','-.y','-.c',':r',':k',':m',':b'};
numcRate_NIDS = length(cRate_NIDS);

cRate_PGEXTRA = [1, 1.2, 1.3, 1.4];
LineSpecs_PGEXTRA = {':b','--b','-b','-m','-c'};
numcRate_PGEXTRA = length(cRate_PGEXTRA);

outputs_NIDS = cell(numcRate_NIDS,1);
outputs_PGEXTRA = cell(numcRate_PGEXTRA,1);

% call methods
for i = 1:numMethods
    paras.method = methods{i};
    
    switch methods{i}
        case {'NIDS'}
            for j = 1:numcRate_NIDS
                cRate = cRate_NIDS(j);
                alpha = cRate./Lips;
                
                c = 1/(1-lambdan)/max(alpha);
                % c = 0.5/max(alpha);
                paras.alpha = alpha;
                paras.c = c;
                outputs_NIDS{j} = obj.minimize(paras);
            end
        case {'PGEXTRA'}
            for j = 1:numcRate_PGEXTRA
                cRate = cRate_PGEXTRA(j);
                
                alpha = cRate./max_Lips*ones(n,1);
                paras.alpha = alpha;
                outputs_PGEXTRA{j}  = obj.minimize(paras);
            end
        otherwise
            display('not set');
    end
end

% plot
legend_lab = cell(numcRate_NIDS+numcRate_PGEXTRA,1);
for j = 1:numcRate_NIDS
    cRate = cRate_NIDS(j);
    legend_lab{j} = ['NIDS-',num2str(cRate),'/L'];
    semilogy(outputs_NIDS{j}.err/norm_x_star,LineSpecs_NIDS{j});
    hold on;
end

for j = 1:numcRate_PGEXTRA
    cRate = cRate_PGEXTRA(j);
    legend_lab{j+numcRate_NIDS} = ['PGEXTRA-',num2str(cRate),'/L'];
    semilogy(outputs_PGEXTRA{j}.err/norm_x_star,LineSpecs_PGEXTRA{j});
    hold on;
end

xlabel('number of iterations');
ylabel('$\frac{\left\Vert \mathbf{x}-\mathbf{x}^{*}\right\Vert}{\left\Vert \mathbf{x}^{*}\right\Vert}$','FontSize',20,'Interpreter','LaTex');

legend(legend_lab,'FontSize',10);

% tol     = 1e-7; xlim([0 20000]); ylim([1e-9 100]);

ylim([4.0E-10 100]);
saveas(h,[resSubPath,'_compa2.fig']);
%             close;

prob.M = M;
prob.x_ori = x_ori;
prob.y_ori = y_ori;
prob.lam = lam;
prob.W = W;

save([resSubPath,'_compa2_prob.mat'],'prob');
end

function a = funGradS(x)
global n p M y_ori
a = zeros(n, p);
for j = 1:n
    a(j,:) = (M(:,:,j)' * (M(:,:,j) * (x(j,:))' - y_ori(:,j)))';
end
end

function a = funR(x)
global n lam
a = 0;
for j = 1:n
    a = a + lam * norm(x(j,:), 1);
end
end

function a = funS(x)
global n M y_ori
a = 0;
for j = 1:n
    a   = a + 0.5 * sum((M(:,:,j) * (x(j,:))' - y_ori(:,j)).^2);
end
end

function a = funProxR(x,t)
global n p lam
a = zeros(n, p);
for j = 1:n
    a(j,:)  = (wthresh(x(j,:), 's', t(j)*lam))';
end
end
