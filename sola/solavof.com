f77 -c main.f deltadj.f parmov.f bc.f pressit.f tms10.f meshset.f vfconv.f petacal.f tilde.f setup.f prtplt.f lavore.f cavovo.f
f77 -o solavof main.o deltadj.o parmov.o bc.o pressit.o tms10.o meshset.o vfconv.o petacal.o tilde.o setup.o prtplt.o lavore.o cavovo.o
