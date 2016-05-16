gr(x; z) :- gr(x; y), gr(y; z)
gr(y; x) :- gr(y; z), gr(z; x)
gr(z; x) :- gr(z; y), gr(y; x)

gr(x; y) :- b(x), a(y)
gr(x; y) :- c(x), b(y)
gr(x; y) :- d(x), c(y)
gr(x; y) :- e(x), d(y)
gr(x; y) :- f(x), e(y)
gr(x; y) :- g(x), f(y)
gr(x; y) :- h(x), g(y)
gr(x; y) :- i(x), h(y)
gr(x; y) :- j(x), i(y)

x,xy yx,y,yz zy,z,zx xz

a(x, y), b(y, x)
