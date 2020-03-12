#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
   }