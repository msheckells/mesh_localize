function qres = quatmultiply(q1,q2)
%Performs Quaternion multiplication of q1*q2. q1 and q2 are nx4 matrices
%q1 is of the form: a1 + b1 i + c1 j + d1 k
qres(:,1) = q1(:,1).*q2(:,1) - q1(:,2).*q2(:,2) - q1(:,3).*q2(:,3) - q1(:,4).*q2(:,4);
qres(:,2) = q1(:,1).*q2(:,2) + q1(:,2).*q2(:,1) + q1(:,3).*q2(:,4) - q1(:,4).*q2(:,3);
qres(:,3) = q1(:,1).*q2(:,3) - q1(:,2).*q2(:,4) + q1(:,3).*q2(:,1) + q1(:,4).*q2(:,2);
qres(:,4) = q1(:,1).*q2(:,4) + q1(:,2).*q2(:,3) - q1(:,3).*q2(:,2) + q1(:,4).*q2(:,1);
end
