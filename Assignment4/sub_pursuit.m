function x=sub_pursuit(b,A,k)
  x=zeros(size(A,2),1);
  [~,ik]=max(abs(A'*b));
  r=b-A(:,ik)*((A(:,ik))\b);
  iter=0;
  while iter<k+1
      [~,ip]=max(abs(A'*r));
      iu=[ik ip];
      x=zeros(size(A,2),1); x(iu)=A(:,iu)\b; 
      [~,ik]=max(abs(x));
       if (norm(b-A(:,ik)*((A(:,ik))\b),2)>norm(r,2))
         break;
       end
       r=b-A(:,ik)*((A(:,ik))\b);
      iter=iter+1;
  end
end