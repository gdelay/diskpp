// This mesh is fitted to varpi only
// varpi can be tuned using leftx (in (0,1) )
// leftx = 0.5
// attention : replace in the text leftx by the chosen value for every use

// Characteristic length of a mesh element
h = 0.045;

// Points
// external square
Point(1) = {0, 0, 0, h};
Point(2) = {leftx , 0, 0, h};
Point(3) = {1, 0, 0, h};
Point(4) = {1, 1, 0, h};
Point(5) = {leftx, 1, 0, h};
Point(6) = {0, 1, 0, h};

// internal points
Point(7) = {leftx, 0.125, 0, h};
Point(8) = {0.875, 0.125, 0, h};
Point(9) = {0.875, 0.875, 0, h};
Point(10) = {leftx, 0.875, 0, h};

// Lines
// external square
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

// internal lines
Line(7) = {2,7};
Line(8) = {7,8};
Line(9) = {8,9};
Line(10)= {9,10};
Line(11)= {10,5};

// Curved loops
Curve Loop(1) = {2,3,4,-11,-10,-9,-8,-7}; // varpi boundary
Curve Loop(2) = {1,7,8,9,10,11,5,6}; // complementary

// Surfaces
Plane Surface(1) = {1};     // varpi
Plane Surface(2) = {2};     // complementary

Physical Surface(1) = {1,2}; // whole domain