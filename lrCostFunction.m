function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= size(X,2);

J = 0;
grad = zeros(size(theta));



sum1=0;
predictions= sigmoid(X*theta);

for i=1:m
  sum1= sum1 + ((y(i,1)*log(predictions(i,1))) + ((1-y(i,1))*log(1-predictions(i,1))));
end
sum=0;

for i=2:n
  sum=sum + (theta(i,1)*theta(i,1));
end
 
J= ((-1/m)*sum1)+ (((lambda)/(2*m))*(sum));





% =============================================================

grad = grad(:);
grad = (1/m)*(((predictions - y)'*X)');

grad(2:n,1)=grad(2:n,1)+ ((lambda/m)*(theta(2:n,1)));

end
