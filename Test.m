clear
rng default
n = 100; 
b = randn(n,1);
 
%First generate Q, the unitary matrix where the multiplication of this
%matrix with it inverse become the identity matrix and this matrix contains
%orthonormal basis of eigenvetors of A
Q = orth(randn(n));
 
% Let define the matrix E as a diagonal matrix where its main diagonal
% matrix contains the orresponding eigenvalues of A
E = diag(abs(randn(n,1)));

% Then the matrix A which is non negative is defined
A = Q*E*Q';

disp("Condition of A is:")
cond(A)
%conditional less than 30 will always work fine
 
x = randn(n,1);
x'*A*x %(non-negative)
 
% problem 1: gradient descent algorithm
%Implement Gradient Descent Algorithm for solving the above optimiza- tion problem.
%Use correctly selected fixed step size Î± and the parameter Î³.
%Draw the experimental convergence rate, i.e., draw the plot log âˆ¥xk âˆ’ xâˆ—âˆ¥2 versus iteration number k, 
%for the algorithm and compare it with the theo- retically predicted one.
 
% iters = 1000; % maximum iterations
tol = 1e-5; % tolerance value
alpha = 1e-2; % step size
 
 
% define objective function
f = 0.5*x'*A*x-b'*x;
 
gradient_norm = 1
 
iters = 0; % inital number of iterations
while iters <1000 && gradient_norm > tol
    % compute gradient 
    gradient = grad_f(x,A,b); % Calculate the gradient
    gradient_norm = norm(gradient); % Gradient Norm
    %fprintf('gradient norm is %f\n',gradient_norm);
    % change x by learning rate alpha
    x_ = x - alpha*gradient;
    iters = iters + 1; % increase by one
    x = x_; % update x
end 
 
x_optimal = x
disp('Loop ends and optimal gradient is:');
disp(gradient_norm);
 
% function grad_f compute gradient of object function at x
function gradient = grad_f(x,A,b)
gradient = x'*(A'+A) - b';
end