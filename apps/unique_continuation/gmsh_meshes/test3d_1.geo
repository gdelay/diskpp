// 3d geometry
// the mesh is fitted to varpi
// attention : this mesh does not work currently ...

// Characteristic length of a mesh element
h = 0.25;

// Points
// external cube
Point(1) = {0, 0, 0, h};
Point(2) = {0, 0, 1, h};
Point(3) = {0, 1, 1, h};
Point(4) = {0, 1, 0, h};
Point(5) = {1, 0, 0, h};
Point(6) = {1, 0, 1, h};
Point(7) = {1, 1, 1, h};
Point(8) = {1, 1, 0, h};


// Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {1,5};
Line(6) = {2,6};
Line(7) = {3,7};
Line(8) = {4,8};

Line(9) = {5,6};
Line(10)= {6,7};
Line(11)= {7,8};
Line(12)= {8,5};


// Surfaces
Line Loop(13) = {1,2,3,4}; // attention sens horaire
Plane Surface(14) = {13};
Line Loop(15) = {1,6,-9,-5};
Plane Surface(16) = {15};
Line Loop(17) = {2,7,-10,-6};
Plane Surface(18) = {17};
Line Loop(19) = {3,8,-11,-7};
Plane Surface(20) = {19};
Line Loop(21) = {4,5,-12,-8};
Plane Surface(22) = {21};
Line Loop(23) = {9,10,11,12};
Plane Surface(24) = {23};


// Volume
Surface Loop(25) = {-13,15,17,19,21,23};
Complex Volume(26) = {25};


// Physical
// Physical Surface(200) = {14,16,18,20,22};