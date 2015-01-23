CFLAGS=
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc
PRGS=$(PRG1) $(PRG2) $(PRG3)

FILTERED = $(foreach num,1 2 3,\
	mer_0$(num)_dx.png mer_0$(num)_dy.png mer_0$(num)_g7.png)

FEATURES = $(foreach num,1 2 3 4 5 6,\
	marshill_image$(num).png marshill_image$(num).txt)

MATCH = $(foreach num,1 2 3 4 5,match_$(num)$(shell bc <<< $(num)\+1).png)

CODE_PRINT = $(foreach f,filters harris match,$(f).pdf)

.PHONY: all codes clean

all: $(FILTERED) $(FEATURES) $(MATCH) gaussian.txt

code_print: $(CODE_PRINT)

mer_%_dx.png: mer_%.bmp filters.py
	python filters.py -q -x -o $@ $<

mer_%_dy.png: mer_%.bmp filters.py
	python filters.py -q -y -o $@ $<

mer_%_g7.png: mer_%.bmp filters.py
	python filters.py -q -g 7 -o $@ $<

marshill_image%.png marshill_image%.txt: marshill_image%.pgm harris.py
	python harris.py -q -o $(subst pgm,png,$<) -t $(subst pgm,txt,$<) $<

match_12.png: \
		marshill_image1.png marshill_image1.txt\
		marshill_image2.png marshill_image2.txt match.py
	python match.py -q -o $@ $(filter-out %.py,$^)

match_23.png: \
		marshill_image2.png marshill_image2.txt\
		marshill_image3.png marshill_image3.txt match.py
	python match.py -q -o $@ $(filter-out %.py,$^)

match_34.png: \
		marshill_image3.png marshill_image3.txt\
		marshill_image4.png marshill_image4.txt match.py
	python match.py -q -o $@ $(filter-out %.py,$^)

match_45.png: \
		marshill_image4.png marshill_image4.txt\
		marshill_image5.png marshill_image5.txt match.py
	python match.py -q -o $@ $(filter-out %.py,$^)

match_56.png: \
		marshill_image5.png marshill_image5.txt\
		marshill_image6.png marshill_image6.txt match.py
	python match.py -q -o $@ $(filter-out %.py,$^)

gaussian.txt: gaussian.py
	python $^ > $@

%.pdf: %.ps
	ps2pdf $^ $@

%.ps: %.py
	vim $^ -c ":hardcopy > $@" -c ":q"

clean:
	rm -f $(FILTERED) $(FEATURES) $(MATCH) *.pdf *.txt
