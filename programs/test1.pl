larger(two, one)
larger(three, two)
larger(four, three)
larger(five, four)
larger(six, five)
larger(seven, six)
larger(eight, seven)
larger(nine, eight)
#larger(ten, nine)
#larger(eleven, ten)
#larger(twelve, eleven)
#larger(thirteen, twelve)
#larger(fourteen, thirteen)
#larger(fifteen, fourteen)
#larger(sixteen, fifteen)
#larger(seventeen, sixteen)
#larger(eighteen, seventeen)
#larger(nineteen, eighteen)
#larger(twenty, nineteen)

larger(X, Z) :- larger(X, Y), larger(Y, Z)
-larger(X, Y) :- larger(Y, X)
-larger(X, X)

larger(four, one)!
larger(three, one)!
larger(four, two)!
-larger(one, two)!
-larger(one, three)!
-larger(one, four)!
-larger(one, five)!
-larger(one, six)!
-larger(one, seven)!
-larger(one, eight)!
-larger(one, nine)!
-larger(two, two)!
-larger(two, three)!
-larger(two, four)!
-larger(two, five)!
-larger(two, six)!
-larger(two, seven)!
-larger(two, eight)!
-larger(two, nine)!
-larger(three, three)!
-larger(three, four)!
-larger(three, five)!
-larger(three, six)!
-larger(three, seven)!
-larger(three, eight)!
-larger(three, nine)!
-larger(four, four)!
-larger(four, five)!
-larger(four, six)!
-larger(four, seven)!
-larger(four, eight)!
-larger(four, nine)!
-larger(five, five)!
-larger(five, six)!
-larger(five, seven)!
-larger(five, eight)!
-larger(five, nine)!
-larger(six, six)!
-larger(six, seven)!
-larger(six, eight)!
-larger(six, nine)!
-larger(seven, seven)!
-larger(seven, eight)!
-larger(seven, nine)!
-larger(eight, eight)!
-larger(eight, nine)!
-larger(nine, nine)!
larger(five, one)!
larger(five, two)!
larger(five, three)!
larger(five, four)!
larger(six, one)!
larger(six, two)!
larger(six, three)!
larger(six, four)!
larger(six, five)!
larger(seven, one)!
larger(seven, two)!
larger(seven, three)!
larger(seven, four)!
larger(seven, five)!
larger(seven, six)!
larger(eight, one)!
larger(eight, two)!
larger(eight, three)!
larger(eight, four)!
larger(eight, five)!
larger(eight, six)!
larger(eight, seven)!
larger(nine, one)!
larger(nine, two)!
larger(nine, three)!
larger(nine, four)!
larger(nine, five)!
larger(nine, six)!
larger(nine, seven)!
