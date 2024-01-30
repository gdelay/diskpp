// This mesh is fitted to varpi only
// varpi is only one quarter of the boundary
// varpi = (0.875,1) x (0,1)
// Characteristic length of a mesh element
h = 0.125;

// Points
// external square
Point(1) = {0, 0, 0, h};
Point(2) = {0.875, 0, 0, h};
Point(3) = {1, 0, 0, h};
Point(4) = {1, 1, 0, h};
Point(5) = {0.875, 1, 0, h};
Point(6) = {0, 1, 0, h};

// Lines
// external square
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

// internal line
Line(7) = {2,5};

// Curved loops
Curve Loop(1) = {2,3,4,-7}; // varpi boundary
Curve Loop(2) = {1,7,5,6}; // complementary

// Surfaces
Plane Surface(1) = {1};     // varpi
Plane Surface(2) = {2};     // complementary

Physical Surface(1) = {1,2}; // whole domain