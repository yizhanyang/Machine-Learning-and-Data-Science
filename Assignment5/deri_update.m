function [hidd, total_energy]= deri_update(obser,hidd,x,y,total_energy,para)
  lam=para(1); beta=para(2);
  if (x>1&&x<size(hidd,1)&&y>1&&y<size(hidd,2))
    hidd(x,y)=1/(1+4*lam+4*beta)*(obser(x,y)+lam*(hidd(x+1,y)+hidd(x-1,y)+hidd(x,y-1)+hidd(x,y+1))+beta*(hidd(x+1,y+1)+hidd(x+1,y-1)+hidd(x-1,y+1)+hidd(x-1,y-1)));
  end
end