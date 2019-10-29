function x=orth_match(b,A,k)
%A=rand(10,50); xx=rand(50,1); b=A*xx; k=10;
    x=zeros(size(A,2),1);
    r=b; i=[]; iter=0;
    while iter<k+1
        [~,ik]=max(abs(A'*r));
        i=[i ik];
        if(norm(b-A(:,i)*(A(:,i)\b),2)>norm(r,2))
          break;
        end
        r=b-A(:,i)*(A(:,i)\b);
        iter=iter+1;
    end
    x(i')=A(:,i)\b;
end