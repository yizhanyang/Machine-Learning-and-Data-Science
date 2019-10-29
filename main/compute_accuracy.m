function  acc= compute_accuracy(t1,y1)
for i=1:size(y1,1)
  y1(i,max(y1(i,:))==y1(i,:))=1;
  y1(i,max(y1(i,:))~=y1(i,:))=0;
end
res=t1-y1;
res(res~=0)=1;
acc=1 - (sum(sum(res~=0))/(2*size(y1,1)));

end

