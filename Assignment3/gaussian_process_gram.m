function [K,C] =gaussian_process_gram(X,theta,precision,T)
    dim= size(X,1);
    K = zeros(dim, dim);
    C=  zeros(dim, dim);
    for i = 1:dim
        for j= 1:i
            K(i,j) =  gaussian_process_kernel(X(i,:), X(j,:), theta);
            K(j,i) = K(i,j);
            if i==j
             C(i,j) = gaussian_process_kernel(X(i,:), X(j,:), theta)+precision;
            else
             C(i,j) = gaussian_process_kernel(X(i,:), X(j,:), theta);
            end
            C(j,i) = C(i,j);
        end      
     end
end