%% ELEC-E5431- Phung Duc Thao – 913223


%% Problem Statement
% In this exercise we are going to slove a optimization problem of a simple
% quadratic function given:
%%
% $$ minarg(x) \frac{1}{2}x^TAx-b^Tx $$
%% Initial parameters

% Clear the console output
clc
% Clear all the variable space
clear
% Set random seed for random generator
rng(860)

n = 100;    
% n = 1000;
% n = 3000;
%% Define Matrix A and b
% First, define variable b where b should be in range of A given dimension
% (n,1)
%%
b = randn(n,1);
disp("The dimension matrix b is:")
size(b)

%%
% Next generate Q, the unitary matrix where the multiplication of this
% matrix with it inverse become the identity matrix and this matrix contains
% orthonormal basis of eigenvetors of A
%%
Q = orth(randn(n));

%%
% Let define the matrix E as a diagonal matrix where its main diagonal
% matrix contains the orresponding eigenvalues of A
%%
E = diag(abs(randn(n,1)));

%%
% Then the matrix A which is non negative is defined as followed:
A = Q*E*Q';
disp("The dimension matrix A is:")
size(A)

%%
% The condition number of matrix A is given as:
%%
disp("Condition of A is:")
condtionNumber = cond(A);
condtionNumber
% conditional less than 30 will always work fine

%%
% Next we define a random or initial variables for x
%%
initial_x = randn(n,1);
disp("The dimension matrix x is:")
size(initial_x)

%%
% Let's check if the x'*A*x is non negative value

%%
disp("The value of x'*A*x is:")
initial_x'*A*initial_x 
%%
% To solve the unconstrained minimization problem, we need to solve a
% system of linear equations Ax=b and the gradient of the objective
% function is in the form Ax-b and it should equal to 0
% The gradient function for the quadratic function is:
%%
% $$ \frac{1}{2}(A'+A)x-b $$
%%
% Since matrix A is non-negative definite, the matrix A is symmetric and A'
% is equal to A. The result gradient function is:
%%
% $$Ax-b$$
%% 
% To get the minimum value for the function, the gradient of function
% should be 0 for optimality, and the gradient is solved as follows:
% Ax-b=0 or Ax=b
%%
Xoptimal=mldivide(A,b);
disp("The size matrix Xoptimal is:")
size(Xoptimal)

%%
% For the above equation, I have tested with different dimensions of x,A
% and b where it appears to be very slow (or sometimes it crashed) (took around 1 min) to solve the
% system linear equation above with dimension n=3000. For the n in range
% 1000 to 2000, my computer was still able to produce a result.

%% Setting parameters for solving problem
% In this section, we are going to solve the optimization problem above
% using different techniques. The same matrix A and b are used for these
% techniques with same maximum iteration number and tolerance value defined
% as following:
%%
x = initial_x; % Capture the initial solution
iterations = 0; % Initial iteration
max_iterations = 1e4; % maximum number of iterations
tol = 1e-5; % tolerance value
% alpha = 1e-2; % step size
% alpha = 0.5e-1; % step size
% alpha = 0.35; % step size
alpha = 1/norm(A); % step size
f = 0.5*(x'*A*x)-(b'*x);
foptimal = 0.5*(Xoptimal'*A*Xoptimal)-(b'*Xoptimal);

gradient_norm = norm(A*x-b); % Define initial gradient norm
log_error = zeros(1,max_iterations);

while iterations < max_iterations && gradient_norm > tol

% First compute the gradient of function
gradient = grad_function(x,A,b); 
% Calculate the norm of the gradient
gradient_norm = norm(gradient); % Gradient Norm
% Update the variable x using the gradient with selected step size
x_ = x - alpha*gradient;
% Decrease the iteration
iterations = iterations + 1; 
% Update the variable x for next iteration
x = x_; 
log_error(iterations+1)=log(norm(x-Xoptimal));
end 
 
x_optimal = x;
disp('Loop ends in the number of iterations:');
disp(iterations);
disp('Norm of optimal gradient is:');
disp(gradient_norm);

% Plotting for Convergence Rate using iteration steps
figure(1)
xaxis = 1:iterations;
yaxis = log_error;
[numRowsX,numColsX] = size(xaxis);
[numRowsY,numColsY] = size(yaxis);
% +2 for removing the initial value for log_error and keep the last element
% + 2 element
rows_to_delete=(numColsX+2:numColsY); % Delete the 0 rows
yaxis(:,rows_to_delete)=[];
yaxis(:,(1))=[];
size(xaxis)
size(yaxis)
plot(xaxis,yaxis)
title("Experimental Convergence Rate")
xlabel("Iteration");
ylabel("log||x-x*||_2");

x = initial_x; % Capture the initial solution
iterations = 0; % Initial iteration
gradient_norm = norm(A*x-b); % Define initial gradient norm
minusGrad = b-A*x;
p = minusGrad;
log_error = zeros(1,max_iterations);


while iterations < max_iterations && gradient_norm > tol
% First compute the gramma of current iteration
alpha=minusGrad'*minusGrad./(p'*A*p);
x = x-alpha*p;
prev_minusGrad = minusGrad;
minusGrad = prev_minusGrad-alpha*A*p;
beta = minusGrad'*minusGrad./prev_minusGrad'*prev_minusGrad;
p = minusGrad+ beta*p;
log_error(iterations+1)=log(norm(x-Xoptimal));
end 

x_optimal = x;
disp('Loop ends in the number of iterations:');
disp(iterations);
disp('Norm of optimal gradient is:');
disp(gradient_norm);

% Plotting for Convergence Rate using iteration steps
figure(2)
xaxis = 1:iterations;
yaxis = log_error;
[numRowsX,numColsX] = size(xaxis);
[numRowsY,numColsY] = size(yaxis);
% +2 for removing the initial value for log_error and keep the last element
% + 2 element
rows_to_delete=(numColsX+2:numColsY); % Delete the 0 rows
yaxis(:,rows_to_delete)=[];
yaxis(:,(1))=[];
size(xaxis)
size(yaxis)
plot(xaxis,yaxis)
title("Experimental Convergence Rate")
xlabel("Iteration");
ylabel("log||x-x*||_2");

%%
% Let define the gradient function
%%
function gradient = grad_function(x,A,b)
    gradient = A*x - b;
end