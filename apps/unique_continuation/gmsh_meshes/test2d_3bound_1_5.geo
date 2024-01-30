// This mesh is fitted to varpi only
// Characteristic length of a mesh element
h = 0.18;

// Points
// external square
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

// internal rectangle
Point(5) = {0, 0.125, 0, h};
Point(6) = {0.875, 0.125, 0, h};
Point(7) = {0.875, 0.875, 0, h};
Point(8) = {0, 0.875, 0, h};

// Lines
// external square
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,8};
Line(5) = {8,5};
Line(6) = {5,1};

// internal rectangle
Line(7) = {5,6};
Line(8) = {6,7};
Line(9) = {7,8};

// Curved loops
Curve Loop(1) = {1,2,3,4,-9,-8,-7,6}; // varpi boundary
Curve Loop(2) = {7,8,9,5}; // complementary

// Surfaces
Plane Surface(1) = {1};     // varpi
Plane Surface(2) = {2};     // complementary

Physical Surface(1) = {1,2}; // whole domain