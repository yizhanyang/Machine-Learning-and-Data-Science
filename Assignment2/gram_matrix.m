function K =gram_matrix(X,sigma)
    dim= size(X,1);
    K = zeros(dim, dim);
    for i = 1:dim
        for j= 1:i
            K(i,j) =  gaussian_kernel(X(i,:), X(j,:), sigma);
            K(j,i) = K(i,j);
        end      
     end
end