# a_lr: 0.1
# a_post: 10000
# a_pre: 1
# a_search: 100
# a_stop: 0.01
# a_stop_search: 1
# c_alpha: 0.0001
# l_alpha: 1e-05
# l_lr: 0.01
# l_n: 100000
# l_stop: 0.01
# research: 1

cat(tom)
cat(tim)
cat(tam)
cat(tup)
cat(top)
-cat(bat)
-cat(rat)
-cat(put)
mouse(jerry)
mouse(mousea)
mouse(mouseb)
mouse(mousec)
-mouse(louse)
human(socrates)
human(socratesa)
human(socratesb)
human(socratesc)
-human(puman)
robot(rob)
robot(roba)
robot(robb)
robot(robc)
likestoeat(a, b)
likestoeat(aa, ba)
likestoeat(ab, bb)
likestoeat(ac, bc)
-likestoeat(na, nb)
-likestoeat(naa, nba)
-likestoeat(nab, nbb)
-likestoeat(nac, nbc)
animal(an)
rodent(ro)
-rodent(nro)
mortal(mo)
-mortal(nmo)
-animal(nan)

rodent(X) :- mouse(X)
-rodent(X) :- cat(X)
animal(X) :- cat(X)
animal(X) :- rodent(X)
-animal(X) :- human(X)
mortal(X) :- human(X)
mortal(X) :- animal(X)
-mortal(X) :- robot(X)
likestoeat(X, Y) :- cat(X), rodent(Y)
-likestoeat(X, Y) :- cat(X), -rodent(Y)
-likestoeat(X, Y) :- mouse(X), animal(Y)
likestoeat(X, Y) :- human(X), animal(Y)

mortal(socrates)!
-mortal(rob)!
mortal(tom)!
mortal(jerry)!
rodent(jerry)!
-rodent(rob)!
-rodent(socrates)!
-rodent(tom)!
rodent(jerry)!
-human(tom)!
-human(jerry)!
-human(rob)!
human(socrates)!
animal(tom)!
animal(jerry)!
-animal(rob)!
-animal(socrates)!
cat(tom)!
-cat(jerry)!
-cat(socrates)!
-cat(rob)!
mouse(jerry)!
-mouse(tom)!
-mouse(socrates)!
-mouse(rob)!
likestoeat(tom, jerry)!
-likestoeat(jerry, tom)!
likestoeat(socrates, tom)!
-likestoeat(jerry, jerry)!
