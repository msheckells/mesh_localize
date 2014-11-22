function qinv = quatinv(q)
%Performs Quaternion q; q is a nx4 matrix
%q is of the form: a1 + b1 i + c1 j + d1 k
qinv = [q(:,1) -q(:,2:4)];
end
