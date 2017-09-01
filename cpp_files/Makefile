CC	= g++
CPPFLAGS	= -std=c++14 -g

OBJS1	= geomagcnv.o
PROG1	= ./geomagcnv

OBJS2	= kpindexcnv.o
PROG2	= ./kpindexcnv

all:	$(PROG1) $(PROG2)

$(PROG1):	$(OBJS1)
	$(CC) $(CPPFLAGS) -o $@ $(OBJS1)

$(PROG2):	$(OBJS2)
	$(CC) $(CPPFLAGS) -o $@ $(OBJS2)

.cpp.o:
	$(CC) $(CPPFLAGS) -c $<

r1:	$(PROG1)
	$(PROG1) 

r2:	$(PROG2)
	$(PROG2) 

.PHONY: clean

clean:
	$(RM) $(OBJS1) $(OBJS2) $(PROG1) $(PROG2)
