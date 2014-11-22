function rpy = quat2rpy(q)
%Converts the quaternion in the form(q0 + q1 i + q2 j + q3 k into the roll
%pitch yaw (ZYX convention) other conventions can be supported in later
%versions. q is nx4 matrix output in radians
rpy(:,1) = atan2(2*(q(:,1).*q(:,2) +q(:,3).*q(:,4)), 1 - 2*(q(:,2).^2 + q(:,3).^2));
rpy(:,2) = asin(2*(q(:,1).*q(:,3) -q(:,4).*q(:,2)));
rpy(:,3) = atan2(2*(q(:,1).*q(:,4) + q(:,2).*q(:,3)), 1 - 2*(q(:,3).^2 + q(:,4).^2));
end