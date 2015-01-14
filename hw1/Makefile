BINDIR=bin
PRG1=$(BINDIR)/load
SRC1=load.cpp
CFLAGS=
LDFLAGS=-lopencv_core -lopencv_highgui
PRGS=$(PRG1)

.PHONY: all clean

all: $(BINDIR) $(PRGS)

$(BINDIR): 
	mkdir $(BINDIR)

$(PRG1): $(SRC1)
	g++ $(LDFLAGS) $(CFLAGS) -o $(PRG1) $^

clean:
	rm $(PRGS)
