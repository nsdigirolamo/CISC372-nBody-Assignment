FLAGS= -DDEBUG -g
LIBS= -lm
ALWAYS_REBUILD=makefile

pnbody: nbody.o pcompute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
pcompute.o: pcompute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
clean:
	rm -f *.o nbody 
