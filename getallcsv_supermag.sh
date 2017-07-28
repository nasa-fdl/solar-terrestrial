#!/bin/bash
#George Gerules NASA FDL 2017 Solar Terrestrial Interactions Team
#20170727 This script works as of today, might not work in the future......
#bash script to download csv geomagnetic data from supermag.jhualp.edu

#dependencies, you have to have curl installed.  There are examples below, if you use wget.
#might be able to download on windows 10, if you have the ubuntu subsystem installed.

#next line is what the download looks like for all 161 stations from the supermag website.
#http://supermag.jhuapl.edu/mag/lib/services/??user=cheung&start=2016-01-01T00:00:00.000Z&interval=23%3A59&service=mag&stations=ALE%2CTHL%2CSVS%2CRES%2CKUV%2CNRD%2CUPN%2CIGC%2CTAL%2CGHC%2CDMH%2CCBB%2CUMQ%2CNAL%2CGDH%2CLYR%2CDNB%2CPGC%2CATU%2CHRN%2CCDC%2CBLC%2CVIZ%2CCNL%2CHOP%2CSTF%2CRAN%2CIQA%2CMCR%2CSCO%2CBJN%2CEKP%2CDED%2CJCO%2CGHB%2CBRW%2CJAN%2CYKC%2CAMK%2CFCC%2CDIK%2CFHB%2CNOR%2CSMI%2CT29%2CFSP%2CFYU%2CRAL%2CSOR%2CT31%2CGIM%2CTRO%2CEAG%2CAND%2CKEV%2CDAW%2CMAS%2CNAQ%2CTIK%2CPBQ%2CKIL%2CPKR%2CPBK%2CAMD%2CCMO%2CLEK%2CABK%2CCHD%2CLRV%2CMUO%2CNAN%2CKIR%2CFMC%2CHLL%2CLOZ%2CT38%2CSOD%2CT33%2CPEL%2CJCK%2CGAK%2CDON%2CT36%2CMEA%2CRVK%2COUL%2CS01%2CLYC%2COUJ%2CPIN%2CFAR%2CT22%2CSIT%2CBRD%2CT37%2CDOB%2CHAN%2CSOL%2CLER%2CNUR%2CSPG%2CUPS%2CKAR%2COTT%2CKVI%2CLOV%2CCLK%2CNEW%2CGML%2CTAR%2CBOR%2CBOX%2CMGD%2CVIC%2CGTF%2CSTJ%2CSHU%2CESK%2CMSH%2CARS%2CBFE%2CROE%2CSBL%2CT25%2CYOR%2CNVS%2CHLP%2CWNG%2CVAL%2CBOU%2CFRD%2CNGK%2CHAD%2CBEL%2CDSO%2CIRT%2CMAB%2CDOU%2CPET%2CT16%2CBDV%2CCLF%2CFUR%2CHRB%2CNCK%2CPAG%2CFRN%2CTHY%2CBSL%2CKHB%2CSUA%2CLON%2CTUC%2CDLR%2CAAA%2CMSR%2CMMB%2CAQU%2CRIK%2CISK%2CDUR%2CBMT%2CEBR%2CSPT%2CPEG%2CLZH%2CCYG%2CKAK%2CTEO%2CSJG%2CSFS%2CONW%2CHTY%2CMID%2CKAG%2CKNY%2CHON%2CJAI%2CCBI%2CSON%2CLNP%2CGUI%2CPHU%2CHYB%2CKOU%2CMUT%2CTAM%2CGUA%2CDLT%2CHUA%2CMBO%2CBNG%2CGAN%2CKTB%2CBIK%2CASC%2CAPI%2CPPT%2CVSS%2CPIL%2CIPM%2CDRW%2CEWA%2CKDU%2CCKI%2CTWN%2CTAN%2CCTA%2CTRW%2CTSU%2CLRM%2CASP%2CHBK%2CPST%2CTDC%2CHER%2CGNG%2CCNB%2CKEP%2CGNA%2CORC%2CLIV%2CAMS%2CEYR%2CCZT%2CHOB%2CPAF%2CVNA%2CMCQ%2CMAW%2CPG4%2CPG3%2CSPA%2CPG1%2CSBA%2CMCM%2CDRV%2CCSY%2CVOS&delta=none&baseline=all&options=+mlt+sza+decl&format=csv

#Next line is just the 14 NSGS geomagnetic web stations.  Put in your own mix of stations
#BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C
#wget  http://supermag.jhuapl.edu/mag/lib/services/??user=cheung&start=2016-01-01T00:00:00.000Z&interval=23%3A59&service=mag&stations=BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C&delta=none&baseline=all&options=+mlt+sza+decl&format=csv

#Next line is just the 14 NSGS geomagnetic web stations.  This is  string if you download through browser.
#http://supermag.jhuapl.edu/mag/lib/services/??user=cheung&start=2016-01-01T00:00:00.000Z&interval=23%3A59&service=mag&stations=BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C&delta=none&baseline=all&options=+mlt+sza+decl&format=csv

#Next line is what you would use to down load one days worth of geomag data from 14 NSGS stations for supermag
#wget -O 2000-01-01_byday.csv --post-data 'user=ggerules&start=2000-01-01T00:00:00.000Z&interval=23%3A59&service=mag&stations=BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C&delta=none&baseline=all&options=+mlt+sza+decl&format=csv' http://supermag.jhuapl.edu/mag/lib/services/

#Next line is what the curl version of downloading one days worth of geomag data from 14 NSGS stations for supermag
#curl -o 2000-01-01_byday.csv --data 'user=cheung&start=2000-01-01T00:00:00.000Z&interval=23%3A59&service=mag&stations=BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C&delta=none&baseline=all&options=+mlt+sza+decl&format=csv' http://supermag.jhuapl.edu/mag/lib/services/


#Next number of lines is building up the command line to download all 14 NSGS geomag data from supermag for the years 2000 to 2017
#Warning: take extra special care to keep the format consistent.  Data will not be downloaded if dates and times are not consistent.
#The actual command is run from bash via the backquote  the pair of ` <-backquotes.  Comment out line below when debugging.  
a="curl --retry 5 --retry-delay 2 --max-time 3600 -o "
mn="01"
d="_byday.csv --data 'user=cheung&start="
e="T00:00:00.000Z&interval=23%3A59&service=mag&stations=BOU%2CBRW%2CBSL%2CCMO%2CDED%2CFRD%2CFRN%2CGUA%2CHON%2CNEW%2CSHU%2CSIT%2CSJG%2CTUC%2C&delta=none&baseline=all&options=+mlt+sza+decl&format=csv' http://supermag.jhuapl.edu/mag/lib/services/"
t1="-"

for yr in `seq 2000 2017`;
do 
  #echo $yr
  for m in {1..12}; do
    if [ $m -le 10 ]; then
      mo=`printf "%02d\n" $m;`
    else
      mo=`printf "%02d\n" $m;`
    fi

    ld=`date -d "$m/1 + 1 month - 1 day" "+%b - %d days" | cut -d" " -f3;` 
    #echo $ld
    #sleep 3   # sleep for 3 seconds to give server possible time to spin up next months data.  Possibly not needed.
    for da in `seq 1 $ld`; do
      if [ $m -le 10 ]; then
        da1=`printf "%02d" $da`
      else
        da1=`printf "%02d" $da`
      fi

       echo $a$yr$t1$mo$t1$da1$d$yr$t1$mo$t1$da1$e   #<--- echoing out the next line
           `$a$yr$t1$mo$t1$da1$d$yr$t1$mo$t1$da1$e`  #<--- actual line that does the work
    done;
  done;
done;



