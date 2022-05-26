#!/bin/bash
#
# Run a thermal plate acquisition
#
# default to seven 100-read ramps
# Require an object name

CAM=n3
NREADS=100
NRAMPS=5
SLEEP=15

usage()
{
    (
        msg=$1
        if test -n "$msg"; then
            echo -e "\nERROR: $msg\n"
        fi
        echo "usage: $0 [-n] [-r] [-s] [-c] testName"
        echo "    -c        - camera name (default $CAM)"
        echo "    -n        - number of ramps (default $NRAMPS)"
        echo "    -r        - number of reads per ramp (default $NREADS)"
        echo "    -s        - seconds to sleep betweem ramps (default $SLEEP)"
        echo "  testName is required, and cannot contain spaces."
    ) >&2
    exit 1
}

while getopts ":c:n:r:s:" opt; do
    case "$opt" in
        c)
            CAM=${OPTARG}
            ;;
        n)
            NRAMPS=${OPTARG}
            (( NRAMPS > 0 && NRAMPS < 100 )) || usage "silly/invalid number of ramps requested: $NRAMPS"
            ;;
        r)
            NREADS=${OPTARG}
            (( NREADS > 1 && NREADS < 301)) || usage "silly/invalid number of reads requested: $NREADS"
            ;;
        s)
            SLEEP=${OPTARG}
            (( SLEEP >= 0 && SLEEP < 1200 )) || usage "silly/invalid sleep time: $SLEEP"
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

testName=$1
if test -z $testName; then
    usage "testName argument is required"
fi

echo "cam=$CAM testName=$testName nramps=$NRAMPS nreads=$NREADS sleep=$SLEEP"

nameFile=/tmp/takeThermal.txt
fnames=""
for i in `seq $NRAMPS`; do
    oneCmd.py hx_$CAM ramp nread=$NREADS exptype=dark objname="$testName" | tee $nameFile
    fname=$(grep filename= $nameFile | sed 's/^[^"]*"//; s/".*//')
    echo "ramp $i of $NRAMPS : fname=$fname"
    fnames="$fnames $fname"
    sleep $SLEEP
done

echo $fnames
