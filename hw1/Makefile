BINDIR=bin
GXX=g++
PRG1=$(BINDIR)/load
PRG2=$(BINDIR)/acquire
SRC1=load.cpp
SRC2=acquire.cpp
CFLAGS=
LDFLAGS=-lopencv_core -lopencv_highgui
PRGS=$(PRG1) $(PRG2)

.PHONY: all clean

all: $(BINDIR) $(PRGS)

$(BINDIR): 
	mkdir $(BINDIR)

$(PRG1): $(SRC1)
	$(GXX) $(LDFLAGS) $(CFLAGS) -o $@ $^

$(PRG2): $(SRC2)
	$(GXX) $(LDFLAGS) $(CFLAGS) -o $@ $^

clean:
	rm $(PRGS)
