%% Header
% File name "IBVS_Monocular.m"
% ELE00142M Machine Vision and Human Machine Interaction
% Lab 3: Visual Servoing with MATLAB

clear
clc
close all

%% Configuration Visualization

% 6R PUMA type robot arm and Body frame (Using Robotics Toolbox)
figure(1)
title('Configuration Visualization')
L(1) = Link([ 000*pi/180  0.65000 0.00000   -090*pi/180    0    000*pi/180]); 
L(2) = Link([ 000*pi/180  0.15000 0.40000    000*pi/180    0    000*pi/180]); 
L(3) = Link([ 000*pi/180 -0.15000 0.00000   -090*pi/180    0   -090*pi/180]); 
L(4) = Link([ 000*pi/180  0.40000 0.00000    090*pi/180    0    000*pi/180]); 
L(5) = Link([ 000*pi/180  0.00000 0.00000   -090*pi/180    0    000*pi/180]); 
L(6) = Link([ 000*pi/180  0.15000 0.00000    000*pi/180    0    000*pi/180]); 
PUMA = SerialLink(L,'name','PUMA');
theta_initial = [045*pi/180  -75*pi/180  075*pi/180  000*pi/180  090*pi/180  000*pi/180];
PUMA.plot(theta_initial,'scale',0.3);

% Space frame
Ts = [1 0 0 0;
      0 1 0 0;
      0 0 1 0;
      0 0 0 1;];
hold on
trplot(Ts,'thick',2,'frame','s','length',0.25,'rgb')

% Task space
xlim([-0.4 1.4])
ylim([-0.4 1.4])
zlim([-0.4 1.4])
length = 1;
height = 0.5; 
x_length = 0:length/10:length;
y_length = 0:length/10:length;
[x_grid,y_grid] = meshgrid(x_length,y_length);
mesh(x_grid,y_grid,height.*ones(11,11));

% Camera frame
% Camera frame is coincident with the body frame
% Do not need visualization

%% Camera calibration

% intrinsic parameter matric of camera
a = [800    0  800   0;
       0  800  800   0;
       0    0    1   0;];

%% Set Target point & 4 interest points

% Target point
X_target = [0.35, 0.35, height + 0.1]';

% Place target point in task space
plot3(X_target(1,1),X_target(2,1),X_target(3,1),'rx')

% 4 interest points in task space
X_s_i_set = [X_target(1,1)-0.05  X_target(1,1)-0.05  X_target(1,1)+0.05   X_target(1,1)+0.05;
         X_target(2,1)-0.05  X_target(2,1)+0.05  X_target(2,1)-0.05   X_target(2,1)+0.05;
         X_target(3,1)-0.1   X_target(3,1)-0.1   X_target(3,1)-0.1    X_target(3,1)-0.1;];

% Place 4 interest points in task space
figure(1)
line(X_s_i_set(1,[1 2]),X_s_i_set(2,[1 2]),X_s_i_set(3,[1 2]),'color','r');
line(X_s_i_set(1,[2 4]),X_s_i_set(2,[2 4]),X_s_i_set(3,[2 4]),'color','r');
line(X_s_i_set(1,[4 3]),X_s_i_set(2,[4 3]),X_s_i_set(3,[4 3]),'color','r');
line(X_s_i_set(1,[3 1]),X_s_i_set(2,[3 1]),X_s_i_set(3,[3 1]),'color','r');
plot3(X_s_i_set(1,1),X_s_i_set(2,1),X_s_i_set(3,1),'ro')
plot3(X_s_i_set(1,2),X_s_i_set(2,2),X_s_i_set(3,2),'ro')
plot3(X_s_i_set(1,3),X_s_i_set(2,3),X_s_i_set(3,3),'ro')
plot3(X_s_i_set(1,4),X_s_i_set(2,4),X_s_i_set(3,4),'ro')
   
%% Desired Parameter for robot control

% Set desired pixel coordinates of 4 interest points
m_c_i_star = [ 400  400 1200 1200;
              1200  400 1200  400;];

% Plot desired position of 4 interest points in camera image
figure(2)
title('Camera Image')
xlim([0 1600])
ylim([0 1600])
hold on
plot(m_c_i_star(1,1),m_c_i_star(2,1),'b*')
plot(m_c_i_star(1,2),m_c_i_star(2,2),'b*')
plot(m_c_i_star(1,3),m_c_i_star(2,3),'b*')
plot(m_c_i_star(1,4),m_c_i_star(2,4),'b*')
line(m_c_i_star(1,[1 2]),m_c_i_star(2,[1 2]),'color','b');
line(m_c_i_star(1,[2 4]),m_c_i_star(2,[2 4]),'color','b');
line(m_c_i_star(1,[4 3]),m_c_i_star(2,[4 3]),'color','b');
line(m_c_i_star(1,[3 1]),m_c_i_star(2,[3 1]),'color','b');

% Calculate desired visual feature 
s_star = Calculate_s(m_c_i_star,a); %equation 2.9

% Set desired Z axis height
z_star = 0.1;

% Termination conditions of iteration
error_max = 0.001;

% Upper limit of angular velocity of robot manipulator joints (rad/s)
joint_velocity_max = 0.5;

% Cameras report 25 images per second
Hz = 25;

% Current angles of robot manipulator joints
theta_current = theta_initial;

% Current pose of robot end-effector
T_s_e_current = PUMA_fkine(theta_current); % This is a 4 by 4 homogenous transformation matrix [Rot Transl; 0 1]

% Set Lambda
Lambda = 10;
% setriteration idex
i = 1;
%% VS Loop Start

while true
    %% Error (i.e. distance between current position and desired position of end-effector)
    % Extract current position from the T_s_e_current 
    % Require a bit of robotics knowledge, see 
    % https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    % if you do not understand the concept of homogenous transformation matrix
    %------ Your code here -----------------------------------------
   
    %---------------------------------------------------------------
    % Calculate the distance between current and target position X_target
    % Construct a stopping condition for the for loop with this error (hint: using error_max)
    % Otherwise THIS LOOP will run FOREVER!!!!
    %------ Your code here -----------------------------------------
   
    %---------------------------------------------------------------
    %% Visual Feature & Visual Feature Error
    
    % Get pixel coordinates of 4 interest points
    T_c_s = inv(T_s_e_current);  
    m_c_i = Camera_3to2(X_s_i_set,T_c_s); 
    
    % Calculate visual feature
    s = Calculate_s(m_c_i,a);
    
    % Calculate visual feature error
    %------ Your code here -----------------------------------------
   
    %---------------------------------------------------------------
    figure(2) % camera image update
    plot(m_c_i(1,1),m_c_i(2,1),'r.')
    plot(m_c_i(1,2),m_c_i(2,2),'r.')
    plot(m_c_i(1,3),m_c_i(2,3),'r.')
    plot(m_c_i(1,4),m_c_i(2,4),'r.')
    if i == 1
        line(m_c_i(1,[1 2]),m_c_i(2,[1 2]),'color','r');
        line(m_c_i(1,[2 4]),m_c_i(2,[2 4]),'color','r');
        line(m_c_i(1,[4 3]),m_c_i(2,[4 3]),'color','r');
        line(m_c_i(1,[3 1]),m_c_i(2,[3 1]),'color','r');
    end
    %% Spatial Velocity Calculation
    % calculate the interaction matrix given the desire z axis value
    Lx = Calculate_Lx(s,z_star); % Your code for Calculate_Lx
    % Spatial velocity
    %------ Your code here, calculate spatial velocity--------------
    % Hint: equation (4) in the paper
    %---------------------------------------------------------------

    %% Inverse Velocity Kinematics of Robot Manipulator
    
    % Angular velocities of robot manipulator joints
    theta_deria = PUMA_ivkine(V_b_e, theta_current, joint_velocity_max);
    
    % Update angles of robot manipulator joints
    theta_current = theta_current + theta_deria*(1/Hz);
    
    % Update pose of robot end-effector
    T_s_e_current = PUMA_fkine(theta_current);
    
    % Plot new pose of end-effector
    figure(1)
    PUMA.plot(theta_current,'scale',0.3);
    i = i + 1;
end


