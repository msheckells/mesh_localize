close all; clear all;

% Found with known measurement
map_scale = .8862;
calibrated_Hmarker_cam =   [0.0120   -0.9998   -0.0107    0.0010;
                            0.9791    0.0096    0.2030    0.2036;
                           -0.2029   -0.0129    0.9791   -0.3100;
                                 0         0         0    1.0000];

% Locations of optitrak markers in map
map_opti_o = [-1.37162; 1.76992; 0.401571];
map_opti_x = [-1.058911; 1.78101; 0.302765];
map_opti_y = [-1.22047; 1.73605; 0.828085];

map_opti_x = map_opti_x - map_opti_o;
map_opti_x = map_opti_x / norm(map_opti_x);
map_opti_y = map_opti_y - map_opti_o;
map_opti_y = map_opti_y / norm(map_opti_y);

map_opti_z = cross(map_opti_x, map_opti_y);
map_opti_z = map_opti_z / norm(map_opti_z);

Rmap_opti = [map_opti_x, map_opti_y, map_opti_z];


% project to rotation space due to imperfect point measurements
[U, sig, V] = svd(Rmap_opti);
Rmap_opti = U*eye(3)*V';

% Qmap_opti = qGetQ(Rmap_opti);
Tmap_opti = map_scale*map_opti_o;

% transform from optitrak to map
Hmap_opti = [Rmap_opti Tmap_opti; 0 0 0 1];

fileNum = 2;
fileNumStr = num2str(fileNum);

% Open and parse data
est_poses = csvread(strcat('data/est_pose', fileNumStr, '.csv'), 1, 3);
true_poses = csvread(strcat('data/true_pose', fileNumStr, '.csv'), 1, 5);

% Open and parse data
fid = fopen(strcat('data/true_pose', fileNumStr, '.csv')); %open file
headers = fgetl(fid);    %get first line
headers = textscan(headers,'%s','delimiter',','); %read first line
format = repmat('%s',1,size(headers{1,1},1)); %count columns and make format string
true_pose_times = textscan(fid,format,'delimiter',','); %read rest of the file
true_pose_times = [true_pose_times{:}];
true_pose_times = [cellfun(@str2num, true_pose_times(:,1:3))];

fid = fopen(strcat('data/est_pose', fileNumStr, '.csv')); %open file
headers = fgetl(fid);    %get first line
headers = textscan(headers,'%s','delimiter',','); %read first line
format = repmat('%s',1,size(headers{1,1},1)); %count columns and make format string
est_pose_times = textscan(fid,format,'delimiter',','); %read rest of the file
est_pose_times = [est_pose_times{:}];
est_pose_times = [cellfun(@str2num, est_pose_times(:,1:3))];

true_poses = [true_pose_times true_poses];
true_poses = true_poses(isfinite(true_poses(:,4)),:);
est_poses = [est_pose_times est_poses];

% Shows one dimension of both data sets to see how they line up
% figure, plot(1:size(true_poses),true_poses(:,4))
% figure, plot(1:size(est_poses),est_poses(:,5))

% Align times by inspection using plots from above
if fileNum == 3
  est_poses(:,3) = est_poses(:,3) - est_poses(137,3);
  true_poses(:,3) = true_poses(:,3) - true_poses(2250,3);
end
if fileNum == 2
  est_poses(:,3) = est_poses(:,3) - est_poses(214,3);
  true_poses(:,3) = true_poses(:,3) - true_poses(2640,3);
end

true_match_idx = matchnearest(est_poses(:,3), true_poses(:,3));

%sample_idx = 60:120;
%sample_idx = 220:300;
sample_idx = 20:450;
cam_est_sample = est_poses(sample_idx,:);

marker_true_sample = true_poses(true_match_idx(sample_idx),:);

% These plots should be time aligned
% figure, plot(1:size(marker_true_sample),marker_true_sample(:,4))
% hold on
% plot(1:size(cam_est_sample),cam_est_sample(:,5), '-g')
% hold off


Tmarker_cam = [];
Qmarker_cam = [];
RPYmarker_cam = [];

est_Topti_cam = [];
est_Zopti_cam = [];
true_Hopti_markers = zeros(4,4,size(sample_idx,2));
true_rpy = [];
for i=1:size(sample_idx,2)
    
    est_pos = map_scale*cam_est_sample(i, 5:7);
    est_rot = qGetR([cam_est_sample(i, 11) cam_est_sample(i, 8:10)]);
    est_Hmap_cam = [est_rot est_pos'; 0 0 0 1];
    
    est_Hopti_cam = Hmap_opti \ est_Hmap_cam;
    est_Topti_cam = [est_Topti_cam est_Hopti_cam(1:3,4)];
    est_Zopti_cam = [est_Zopti_cam est_Hopti_cam(1:3,1)];
    
    %Hopti_cam_quat = qGetQ(est_Hcam_opti(1:3, 1:3));  
    %Hopti_cam_rpy = quat2rpy([Hopti_cam_quat(4), Hopti_cam_quat(1:3)'])*180/pi;
    
    true_pos = marker_true_sample(i, 4:6);
    true_rot = qGetR([marker_true_sample(i, 10) marker_true_sample(i, 7:9)]);
    true_rpy = [true_rpy quat2rpy([marker_true_sample(i, 10), marker_true_sample(i, 7:9)])'];
    %true_quat = marker_true_sample(i, 7:10);
    %true_rpy = quat2rpy([true_quat(4), true_quat(1:3)])*180/pi;
    
    true_Hopti_marker = [true_rot true_pos'; 0 0 0 1];
    true_Hopti_markers(:,:,i) = true_Hopti_marker;
    
    Hmarker_cam =  true_Hopti_marker \ est_Hopti_cam;
    
    Hmarker_cam_quat = qGetQ(Hmarker_cam(1:3, 1:3));
    Hmarker_cam_rpy = quat2rpy(Hmarker_cam_quat(1:4)')';
    
    RPYmarker_cam = [RPYmarker_cam Hmarker_cam_rpy];
    Qmarker_cam = [Qmarker_cam qGetQ(Hmarker_cam(1:3, 1:3))];
    Tmarker_cam = [Tmarker_cam Hmarker_cam(1:3, 4)];
end


mean_Tmarker_cam = mean(Tmarker_cam');
mean_Qmarker_cam = mean(Qmarker_cam');
mean_Rmarker_cam = qGetR(mean_Qmarker_cam);
mean_Hmarker_cam = [mean_Rmarker_cam mean_Tmarker_cam'; 0 0 0 1];

disp(['Hmarker_cam info:']);
disp(['transalation mean: ', num2str(mean_Tmarker_cam)]);
disp(['translation variance: ', num2str(var(Tmarker_cam'))]);

disp(['quat mean: ', num2str(mean_Qmarker_cam)]);
disp(['quat variance: ', num2str(var(Qmarker_cam'))]);

disp(['rpy mean: ', num2str(mean(unwrap(RPYmarker_cam')*180/pi))]);
disp(['rpy variance: ', num2str(var(unwrap(RPYmarker_cam')*180/pi))]);


figure
quiver3(est_Topti_cam(1,:), est_Topti_cam(2,:), est_Topti_cam(3,:), est_Zopti_cam(1,:), est_Zopti_cam(2,:), est_Zopti_cam(3,:),'AutoScale','off')
axis([-.5 2 -.5 2 0 2])
xlabel('x')
ylabel('y')


true_Hopti_cams = zeros(4,4,size(sample_idx,2));
for i=1:size(sample_idx,2)
    true_Hopti_cams(:,:,i) = true_Hopti_markers(:,:,i)*calibrated_Hmarker_cam;
end

true_opti_cams_x = reshape(true_Hopti_cams(1,4,:),[size(true_Hopti_cams,3) 1]);
true_opti_cams_y = reshape(true_Hopti_cams(2,4,:),[size(true_Hopti_cams,3) 1]);
true_opti_cams_z = reshape(true_Hopti_cams(3,4,:),[size(true_Hopti_cams,3) 1]);

figure, plot(est_Topti_cam(1,:), est_Topti_cam(2,:), '-g')
hold on
plot(true_opti_cams_x,true_opti_cams_y)

errors = sqrt(sum((est_Topti_cam(1:3,:) - [true_opti_cams_x'; true_opti_cams_y'; true_opti_cams_z']).^2, 1)); 

mean_error = mean(errors);
var_error = var(errors);

disp(['--------------------------', char(13), 'Error Mean: ', num2str(mean_error)]);
disp(['Error Variance: ', num2str(var_error)]);


