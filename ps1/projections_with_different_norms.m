%% For y(1)
[y1, r1] = proj_cvx([3;2], [-2;-4], [-2;5], 1)

%% For y(inf)
[yinf, rinf] = proj_cvx([3;2], [-2;-4], [-2;5], inf)

%% For y(2)
[y2, r2] = proj_cvx([3;2], [-2;-4], [-2;5], 2)

function [y, r] = proj_cvx(x, v0, v, nrm)%% x, v0 and v must be  column  vectors
    objtv = @(y) norm(x-y, nrm);%% objective  is L_2  norm
    cvx_begin
        variable y(2)%% 2-d variable  we are  optimizing  over
        variable t(1)%% real  valued  parameter  that  defines
        minimize(objtv(y))%% defining  the  objective
        subject  to
        v0 + t*v == y%% the  projection y must be in set A
    cvx_end
    r = objtv(y);%% minimum  value of the  objective
end
