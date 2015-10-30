#!/bin/bash

FILENAME="lrvb_nips"

Rscript -e 'library(knitr); knit("'$FILENAME'.rnw")'
bibtex ${FILENAME}.aux
pdflatex ${FILENAME}.tex
#evince ${FILENAME}.pdf
