function vecres = quatrotate(q,vec)
sizevec = size(vec,1);%Number of rows
qvec = [zeros(sizevec,1) vec];
vecres = quatmultiply(q, quatmultiply(qvec, quatinv(q)));
vecres(:,1) = [];
end
