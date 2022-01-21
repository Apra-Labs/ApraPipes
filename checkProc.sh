#!/bin/bash
pCnt=$(nproc);
thCnt=$(($pCnt/2));
if [ $thCnt -lt 1 ]; then
    thCnt=1;
fi
echo $thCnt
