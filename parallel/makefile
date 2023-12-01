CC = nvcc
SDIR = sources
IDIR = headers
ODIR = objects

OBJS += $(ODIR)/nbody.o
OBJS += $(ODIR)/compute.o

CFLAGS = -I $(IDIR) -DDEBUG -DALT_PRINT_SYSTEM # -DKERNEL_ARGS_DEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile

run: nbody
	srun ./nbody > out.txt

nbody: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(ODIR)/%.o: $(SDIR)/%.cu | $(ODIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(ODIR):
	mkdir -p $@

clean:
	rm -f $(OBJS) nbody out.txt
