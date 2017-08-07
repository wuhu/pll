larger(two, one)
larger(three, two)
larger(four, three)
larger(X, Z) :- larger(X, Y), larger(Y, Z)
-larger(X, Y) :- larger(Y, X)

larger(four, one)!
larger(three, one)!
larger(four, two)!
-larger(one, two)!
-larger(one, three)!
-larger(one, four)!
-larger(two, three)!
-larger(two, four)!
-larger(three, four)!

