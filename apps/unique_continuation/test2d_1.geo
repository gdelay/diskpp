// This mesh is fitted to B and varpi
// Characteristic length of a mesh element
h = 0.25;

// Points
// external square
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

// internal square
Point(5) = {0.125, 0.125, 0, h};
Point(6) = {0.875, 0.125, 0, h};
Point(7) = {0.875, 0.875, 0, h};
Point(8) = {0.125, 0.875, 0, h};

// additional points
Point(9) = {0, 0.875, 0, h};
Point(10)= {0, 0.125, 0, h};

// Lines
// external square
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,9};
Line(5) = {9,10};
Line(6) = {10,1};

// internal square
Line(7) = {5,6};
Line(8) = {6,7};
Line(9) = {7,8};
Line(10) = {8,5};

// additional lines
Line(11) = {9,8};
Line(12) = {10,5};

// Curved loops
Curve Loop(1) = {1,2,3,4,11,-9,-8,-7,-12,6}; // varpi boundary
Curve Loop(2) = {7,8,9,10}; // B boundary
Curve Loop(3) = {12,-10,-11,5}; // complementary

// Surfaces
Plane Surface(1) = {1};     // varpi
Plane Surface(2) = {2};     // B
Plane Surface(3) = {3};     // complementary

Physical Surface(1) = {1,2,3}; // whole domain