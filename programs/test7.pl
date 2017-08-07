cat(tom)
mouse(jerry)
rodent(X) :- mouse(X)
-rodent(X) :- cat(X)
likestoeat(X, Y) :- cat(X), rodent(Y)
-likestoeat(X, Y) :- cat(X), -rodent(Y)

likestoeat(a,b)
-likestoeat(b,c)

rodent(jerry)!
likestoeat(tom, jerry)!
-likestoeat(tom, tom)!
