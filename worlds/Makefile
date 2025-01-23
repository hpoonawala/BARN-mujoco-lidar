
WS := $(wildcard *.world)
CSVS := $(patsubst %.world,%.csv,$(WS))

# $(info $(MDS))

all: $(CSVS)	

# won't do anything if the pdfs exist
$(CSVS): %.csv : %.world
	nvim -c '/state' -c ':norm dat' -c '/gui' -c ':norm dat' -c 'v/pose/ :delete ' -c 'g/pos/ :norm dst' -c ':1 :delete' -c ':saveas %:r.csv' -c 'qa' $<
	nvim -c ':% :norm <<' -c ':% :norm <<' -c '%s/ /,/g' -c ':% :norm A,' -c ':saveas %:r.csv' -c 'qa' $@
