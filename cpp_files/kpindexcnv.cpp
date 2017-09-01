#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace std;

//############################### class CStation #######################
class CStation
{
public:
  CStation()
  { m_nAndx = 0; 
    m_nKP03 = 0;
    m_nKP06 = 0;
    m_nKP09 = 0;
    m_nKP12 = 0;
    m_nKP15 = 0;
    m_nKP18 = 0;
    m_nKP21 = 0;
    m_nKP24 = 0;
  }

  CStation(string sDate,
           string sStation,
           string sLat,
           string sLong,
           int    nAndx,
           int    nKP03,
           int    nKP06,
           int    nKP09,
           int    nKP12,
           int    nKP15,
           int    nKP18,
           int    nKP21,
           int    nKP24 )
  { m_sDate = sDate;
    m_sStation = sStation;
    m_sLat  = sLat;
    m_sLong = sLong;
    m_nAndx = nAndx;
    m_nKP03 = nKP03;
    m_nKP06 = nKP06;
    m_nKP09 = nKP09;
    m_nKP12 = nKP12;
    m_nKP15 = nKP15;
    m_nKP18 = nKP18;
    m_nKP21 = nKP21;
    m_nKP24 = nKP24;
  }

  // copy constructor
  CStation( const CStation& b ) 
  { m_sDate = b.m_sDate;
    m_sStation = b.m_sStation;
    m_sLat  = b.m_sLat;
    m_sLong = b.m_sLong;
    m_nAndx = b.m_nAndx; 
    m_nKP03 = b.m_nKP03;
    m_nKP06 = b.m_nKP06;
    m_nKP09 = b.m_nKP09;
    m_nKP12 = b.m_nKP12;
    m_nKP15 = b.m_nKP15;
    m_nKP18 = b.m_nKP18;
    m_nKP21 = b.m_nKP21;
    m_nKP24 = b.m_nKP24;
  }

  // assignment
  CStation& operator=( const CStation& b ) 
  { m_sDate = b.m_sDate;
    m_sStation = b.m_sStation;
    m_sLat  = b.m_sLat;
    m_sLong = b.m_sLong;
    m_nAndx = b.m_nAndx; 
    m_nKP03 = b.m_nKP03;
    m_nKP06 = b.m_nKP06;
    m_nKP09 = b.m_nKP09;
    m_nKP12 = b.m_nKP12;
    m_nKP15 = b.m_nKP15;
    m_nKP18 = b.m_nKP18;
    m_nKP21 = b.m_nKP21;
    m_nKP24 = b.m_nKP24;
    return *this;
  }  

  string m_sDate;
  string m_sStation;
  string m_sLat;
  string m_sLong;
  int    m_nAndx;  
  int    m_nKP03;
  int    m_nKP06;
  int    m_nKP09;
  int    m_nKP12;
  int    m_nKP15;
  int    m_nKP18;
  int    m_nKP21;
  int    m_nKP24;
};

// description of columns of data
//   also example data
#if 0 
:Product: Geomagnetic Data             201001AK.txt
:Issued: 1637 UTC 28 Feb 2010
#
# Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center
# Please send comments and suggestions to SWPC.Webmaster@noaa.gov
# Updated daily. Values are shown as reported, SEC does not verify accuracy.
# Missing Data: -1
#
#      Geomagnetic A and K indices from the U.S. Geological Survey Stations
#
#               Geomagnetic
#                 Dipole     A   ------------- 3 Hourly K Indices --------------
# Station        Lat Long  Index 00-03 03-06 06-09 09-12 12-15 15-18 18-21 21-24
#-------------------------------------------------------------------------------

2010 Jan 1

Beijing          N29 W174   -1    -1    -1    -1    -1    -1    -1    -1    -1
Belsk            N69 E 40    5     1     1     1     1     1     2     2     2
Boulder          N49 W 42    1     0     0     0     0     0     1     0     2
Cape Chelyuskin  N66 E177   -1    -1    -1    -1    -1    -1    -1    -1    -1
Chambon-la-foret N-- E---   -1    -1    -1    -1    -1    -1    -1    -1    -1
College          N65 W102    0     0     0     0     0     0     0     0     0
Dixon Island     N63 E162    5     0     0     0     0     0     0     1     1
Fredericksburg   N38 W 78    2     0     0     0     0     0     1     1     2
Gottingen(provisional Ap)   -1    -1    -1    -1    -1    -1    -1    -1    -1
Kergulen Island  S57 E130   -1    -1    -1    -1    -1    -1    -1    -1    -1
Krenkel          N71 E156   -1    -1    -1    -1    -1    -1    -1    -1    -1
Learmonth        S22 E114   -1    -1    -1    -1    -1    -1    -1    -1    -1
St. Petersburg   N56 E118    3     1     1     1     1     1     1     1     1
Magadan          N51 W148    0     0     0     0     0     0     0     0     1
Moscow           N51 E122    5     2     2     2     2     2     1     0     0
Murmansk         N63 E127   -1    -1    -1    -1    -1    -1    -1    -1    -1
Novosibirsk      N45 E159    0    -1     0     0     0     0     0     0     1
P. Tunguska      N51 E165   -1    -1    -1    -1    -1    -1    -1    -1    -1
Petropavlovsk    N45 W140   -1    -1    -1    -1    -1    -1    -1    -1    -1
Planetary(estimated Ap)      1     1     0     0     0     0     0     3     1
Tiksi Bay        N61 W168   -1    -1    -1    -1    -1    -1    -1    -1    -1
Wingst           N54 E 95   -1    -1    -1    -1    -1    -1    -1    -1    -1

#endif


int main()
{
  cout << "NASA FDL kpindex conversion program\n";

  ifstream inflst; //list of ak txt files 
  ifstream inStat; //stations
  ifstream invtxt; // a csv file
  string fn;

  //               Geomagnetic
  //                 Dipole     A   ------------- 3 Hourly K Indices --------------
  //Station        Lat Long  Index 00-03 03-06 06-09 09-12 12-15 15-18 18-21 21-24
  string sDate;
  string sStation;
  string sLat;
  string sLong;
  int    nAndx = 0;  
  int    nKP03 = 0;
  int    nKP06 = 0;
  int    nKP09 = 0;
  int    nKP12 = 0;
  int    nKP15 = 0;
  int    nKP18 = 0;
  int    nKP21 = 0;
  int    nKP24 = 0;
  bool   bPdate = true;

  vector<string> arrFName;
  string ln;
  char szBuff[1000];

  vector<CStation> arrSt;

  inflst.open("ak_filelist.out");
  if(inflst)
  {
    cout << "Opened filelist.out\n";
    while(inflst >> fn)
    {
      //cout << fn << endl;
      arrFName.push_back(fn);
    }
    inflst.close();
  }
  else
  {
    cout << "filelist.out file not found\n";
    return 0;
  }

  long int sz = arrFName.size();
  long int cnt = 0;
  long int rcnt = 0;
  char* pStr = NULL;
  string tln;
  string tfn;
  int  bcnt = 0;
  int  ccnt = 0;
  for(unsigned int i = 0; i < arrFName.size(); i++  )
  {
    fn = arrFName[i]; 
    tfn = fn;
    cout << i << " " << fn << endl;
    invtxt.open(fn);
    if(invtxt)
    {

      while(!invtxt.eof()) 
      {
        getline(invtxt, ln);
        tln = ln;
        if(tln.empty())
        {
          rcnt++;         
          continue;
        }
        // skip header record stuff
        if(tln[0] == ':')
        {
          rcnt++;         
          continue;
        }
        if(tln[0] == '#')
        {
          rcnt++;         
          continue;
        }

        if(bPdate)
        {
          sDate = tln;
          bPdate = false; 
          bcnt = 0;
          sDate = tln;
        }
        else
        {
          if(bcnt == 21)
          {
            bPdate = true;
            continue;
          }
          bcnt++;

          // peel apart the record
          // gwg store stuff in CStation
          strcpy(szBuff, tln.c_str()); 
          pStr = strtok(szBuff, " \n");
          while(pStr)
          {
            string s1(pStr);
            sStation = s1;

            pStr = strtok(NULL, " \n");
            string s2(pStr);
            sLat = s2;

            pStr = strtok(NULL, " \n");
            string s3(pStr);
            sLong = s3;

            pStr = strtok(NULL, " \n");
            nAndx = atoi(pStr);  
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP03 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP06 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP09 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP12 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP15 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP18 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP21 = atoi(pStr);
            ccnt++;

            pStr = strtok(NULL, " \n");
            nKP24 = atoi(pStr);
            ccnt++;

            if(ccnt == 9)
            {
              CStation cst;
              cst.m_sDate = sDate;
              cst.m_sStation = sStation;
              cst.m_sLat = sLat;
              cst.m_sLong = sLong;
              cst.m_nAndx = nAndx;
              cst.m_nKP03 = nKP03;
              cst.m_nKP06 = nKP06;
              cst.m_nKP09 = nKP09;
              cst.m_nKP12 = nKP12;
              cst.m_nKP15 = nKP15;
              cst.m_nKP18 = nKP18;
              cst.m_nKP21 = nKP21;
              cst.m_nKP24 = nKP24;

              arrSt.push_back(cst);

              pStr = strtok(NULL, " \n");
              if(pStr == NULL)
                break;
            }
          }
        }

        cnt++;         
        rcnt++;         
      }
      invtxt.close();
    }
    else
    {
      cout << "Error\n";
    }
  }
  cout << "Number of records processed -> " << rcnt << endl;

  cout << "Done!\n";

  return 0;
}

