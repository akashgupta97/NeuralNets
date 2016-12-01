function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


p = zeros(size(X, 1), 1);

X=[ones(m,1) X];
a1= sigmoid((Theta1* X')');
a1=[ones(m,1) a1];
a2= sigmoid((Theta2 * a1')');
[M,I]=max(a2,[],2);
p(:)=I(:);
 

% =========================================================================


end
