function idx = matchnearest2(tt1, tt2, maxdt)
% MATCHNEAREST2 - Find unique matching events in two time series
%    MATCHNEAREST2 is just like MATCHNEAREST2, except that it guarantees
%    that the matches are unique, that is, a given event in TT2 can be matched
%    to at most one event in TT1 (and vice versa).

if nargin<3
  maxdt = inf;
end

idx = matchnearest(tt1, tt2, maxdt);
backidx = matchnearest(tt2, tt1, maxdt);

N1=length(tt1);
N2=length(tt2);
idx(idx==0) = N2+1;
backidx(N2+1)=0;
id=[1:N1]';
mtch = backidx(idx);
idx(mtch~=id)=0;
