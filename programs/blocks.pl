above(X, Y) :- on(X, Y)
above(X, Y) :- above(X, Z), above(Z, Y)

below(Y, X) :- above(X, Y)
below(X, Y) :- -above(X, Y)

stack(X, Y) :- above(X, Y), on_ground(Y)

left_of(X, Y) :- left_of(X, Z), left_of(Z, Y)
left_of(X, Y) :- stack(X, Z), stack(Y, W), left_of(Z, W)
right_of(X, Y) :- left_of(Y, X)
right_of(X, Y) :- -left_of(X, Y)

