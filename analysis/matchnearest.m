function idx = matchnearest(tt1, tt2, maxdt)
% MATCHNEAREST - Find matching events in two time series
%   idx = MATCHNEAREST(tt1, tt2) returns a vector in which the k-th
%   element indicates which event in time series TT2 occured most
%   closely to the k-th event in time series TT1.
%   idx = MATCHNEAREST(tt1, tt2, maxdt) specifies a maximum time interval
%   beyond which matches cannot be declared.
%   Events that do not have a match result in a zero entry in the IDX.
%   Note that this function does not guarantee that the matching is
%   one-to-one, i.e., it is possible for more than one event in TT1 to be
%   matched against a given event in TT2. See MATCHNEAREST2 if this is
%   undesirable.

if nargin<3
  maxdt = inf;
end

N = length(tt1);
idx = zeros(N,1);
for n=1:N
  t0 = tt1(n);
  dt = tt2 - t0;
  [mindt, id] = min(abs(dt));
  if mindt<maxdt
    idx(n) = id;
  end
end
