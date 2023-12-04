FLAGS = -DDEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile

run: nbody
	./nbody > out.txt

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody pnbody out.txt pout.txt
