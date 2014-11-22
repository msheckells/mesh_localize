function q = rpy2quat(rpy)
%Converts rpy body 321 sequence (yaw pitch roll) to quaternion
%q = q0 + q1 i + q2 j + q3 k 
r = rpy(3);
p = rpy(2);
y = rpy(1);
q = [cos(r/2)*cos(p/2)*cos(y/2) + sin(r/2)*sin(p/2)*sin(y/2);
     sin(r/2)*cos(p/2)*cos(y/2) - cos(r/2)*sin(p/2)*sin(y/2);
     cos(r/2)*sin(p/2)*cos(y/2) + sin(r/2)*cos(p/2)*sin(y/2);
     cos(r/2)*cos(p/2)*sin(y/2) - sin(r/2)*sin(p/2)*cos(y/2)];
end