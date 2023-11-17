function Lx = Calculate_Lx(s,Z)
    % Function to calculate the interaction matrix or image Jacobian
    % hint: Eq (11) on the paper or on slide 21 assuming focal length = 1
    % Inputs :
    % s: features
    % z: distance along z axis
    % Output:
    % Lx: interaction matrix/image Jacobian
    % Your code here
    Lx1 = [
            -1/Z, 0   , s(1)/Z, s(1)*s(2)   , -(1 + (s(1)^2)), s(2);
            0   , -1/Z, s(2)/Z, 1 + (s(2)^2), -s(1)*s(2)     , -s(1)
          ];
    Lx2 = [
            -1/Z, 0   , s(3)/Z, s(3)*s(4)   , -(1 + (s(3)^2)), s(4);
            0   , -1/Z, s(4)/Z, 1 + (s(4)^2), -s(3)*s(4)     , -s(3)
          ];
    Lx3 = [
            -1/Z, 0   , s(5)/Z, s(5)*s(6)   , -(1 + (s(5)^2)), s(6);
            0   , -1/Z, s(6)/Z, 1 + (s(6)^2), -s(5)*s(6)     , -s(5)
          ];
    Lx4 = [
            -1/Z, 0   , s(7)/Z, s(7)*s(8)   , -(1 + (s(7)^2)), s(8);
            0   , -1/Z, s(8)/Z, 1 + (s(8)^2), -s(7)*s(8)     , -s(7)
          ];
    Lx = [Lx1;Lx2;Lx3;Lx4];
end