# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples

FLAGS= -g
LIBS= -lm
ALWAYS_REBUILD=makefile

pnbody: pnbody.o pcompute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
pnbody.o: pnbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
pcompute.o: pcompute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody pnbody output.txt poutput.txt
