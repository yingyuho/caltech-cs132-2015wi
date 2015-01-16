BINDIR=bin
GXX=g++

PRG1=$(BINDIR)/load
SRC1=load.cpp

PRG2=$(BINDIR)/acquire
SRC2=acquire.cpp

PRG3=$(BINDIR)/calib
SRC3=calib.cpp

CFLAGS=
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc
PRGS=$(PRG1) $(PRG2) $(PRG3)

.PHONY: all clean

all: $(BINDIR) $(PRGS)

$(BINDIR): 
	mkdir $(BINDIR)

$(PRG1): $(SRC1)
	$(GXX) $(LDFLAGS) $(CFLAGS) -o $@ $^

$(PRG2): $(SRC2)
	$(GXX) $(LDFLAGS) $(CFLAGS) -o $@ $^

$(PRG3): $(SRC3)
	$(GXX) $(LDFLAGS) $(CFLAGS) -o $@ $^

clean:
	rm $(PRGS)
