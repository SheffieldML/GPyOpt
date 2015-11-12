close all
clear

x = [1 -1 -1 1 1];
y = [3 3 1 1 3];

mux = 2*rand-1;
muy = 2*rand-1;

th = 2*pi*rand;

s1 = 1/rand;
s2 = 1/rand;
% s1 = 10^-2;
% s2 = 1;

scalesmat = diag([s1,s2]);
covmat = scalesmat*[1,cos(th);cos(th),1]*scalesmat;

covmat = [1 1;1 2];

pq = [cos(th) sin(th);-sin(th) cos(th)]*[x-mux;y-muy];
p = pq(1,:);
q = pq(2,:);

uv = scalesmat^(-1)*pq;
u = uv(1,:);
v = uv(2,:);

uvd = [cos(th) -sin(th);sin(th) cos(th)]*uv;
ud = uvd(1,:);
vd = uvd(2,:);

pqd = scalesmat*uvd;
pd = pqd(1,:);
qd = pqd(2,:);

trans_mat = scalesmat*[cos(th) -sin(th);sin(th) cos(th)]*scalesmat^(-1)*[cos(th) sin(th);-sin(th) cos(th)]
inv_trans_mat = inv(trans_mat);

aup = mean(pd([1,4]));
adp = mean(pd([2,3]));
auq = mean(qd([1,2]));
adq = mean(qd([3,4]));

ap = [aup,adp,adp,aup,aup];
aq = [auq,auq,adq,adq,auq];

axy = inv_trans_mat*[ap;aq];
ax = axy(1,:) + mux;
ay = axy(2,:) + muy;

low = -8;
high = 8;

xvec = linspace(low,high,30);%min([pd,x]),max([pd,x]),30);
yvec = linspace(low,high,30);%min([qd,y]),max([qd,y]),30);
[X,Y] = meshgrid(xvec,yvec)

F = nan(length(xvec),length(yvec));
G = nan(length(xvec),length(yvec));
for i=1:length(xvec)
    for j=1:length(yvec)
        F(j,i) = mvnpdf([xvec(i);yvec(j)],[mux;muy],covmat);
        G(j,i) = mvnpdf([xvec(i);yvec(j)],[0;0],scalesmat.^2);
    end
end

figure;

xlabel x
ylabel y
hold on;
contourf(X,Y,F);
plot(x,y,'w','LineWidth',2);
plot(mean(x(1:4)),mean(y(1:4)),'w+')
plot(ax,ay,'k','LineWidth',2);
axis square;
axis([low,high,low,high]);

figure;

xlabel p
ylabel q
hold on;
% plot(p,q,'r')
% plot(u,v,'g')
% plot(ud,vd,'b')
contourf(X,Y,G);
plot(pd,qd,'w','LineWidth',2);
plot(mean(pd(1:4)),mean(qd(1:4)),'w+')
plot(ap,aq,'k','LineWidth',2);
axis square;
axis([low,high,low,high]);
