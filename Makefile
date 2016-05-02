#
#  Makefile for campus Linux computers
#

INCLUDE = -I/usr/include -I/usr/X11/include
LIBDIR = -L/usr/local/lib
LIBS =  -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_core -lopencv_video

CFLAGS = -g

CC = g++

all: ballDetector_kalman

ALL.O = ballDetector_kalman.o

ALL.CPP = $(subst .o,.cpp,$(ALL.O)) 

MAKEDEPEND = gcc -M $(CPPFLAGS) $(INCLUDE) -o $*.d $<

%.P : %.cpp
	$(MAKEDEPEND)
	@sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' < $*.d > $@; \
		rm -f $*.d; [ -s $@ ] || rm -f $@

-include $(ALL.O:.o=.P)

%.o: %.cpp 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

ballDetector_kalman: $(ALL.O)
	$(CC) $(CFLAGS) -o $@ $(ALL.O) $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm -f *.o *.P ballDetector_kalman *~ ._*
