#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>

using namespace std;

int main()
{
  cout << "NASA FDL geomag conversion program\n";

  ifstream inflst; //list of csv files 
  ifstream inStat; //stations
  ifstream infcsv; // a csv file
  string fn;

  //Date_UTC,IAGA,MLT,MLAT,IGRF_DECL,SZA,N,E,Z
  string sDate_UTC;
  string sIAGA;
  float  fMLT = 0.0;
  float  fMLAT = 0.0;
  float  fIGRF_DECL = 0.0;
  float  fSZA = 0.0;
  float  fN = 0.0;
  float  fE = 0.0;
  float  fZ = 0.0;
  vector<string> arrFName;
  string ln;
  char szBuff[1000];

  vector<string> arrBOU;
  vector<string> arrBRW;
  vector<string> arrBSL;
  vector<string> arrCMO;
  vector<string> arrDED;
  vector<string> arrFRD;
  vector<string> arrFRN;
  vector<string> arrGUA;
  vector<string> arrHON;
  vector<string> arrNEW;
  vector<string> arrSHU;
  vector<string> arrSIT;
  vector<string> arrSJG;
  vector<string> arrTUC;

  ofstream ofBOU;
  ofstream ofBRW;
  ofstream ofBSL;
  ofstream ofCMO;
  ofstream ofDED;
  ofstream ofFRD;
  ofstream ofFRN;
  ofstream ofGUA;
  ofstream ofHON;
  ofstream ofNEW;
  ofstream ofSHU;
  ofstream ofSIT;
  ofstream ofSJG;
  ofstream ofTUC;

  //Stations
  //BOU BRW BSL CMO DED FRD FRN GUA HON NEW SHU SIT SJG TUC

  long nBOU = 0;
  long nBRW = 0;
  long nBSL = 0;
  long nCMO = 0;
  long nDED = 0;
  long nFRD = 0;
  long nFRN = 0;
  long nGUA = 0;
  long nHON = 0;
  long nNEW = 0;
  long nSHU = 0;
  long nSIT = 0;
  long nSJG = 0;
  long nTUC= 0;

  inflst.open("filelist.out");
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
  
  cout << "CSV File Reading\n";
  long int sz = arrFName.size();
  long int cnt = 0;
  long int rcnt = 0;
  char* pStr = NULL;
  string tln;
  string tfn;
  for(unsigned int i = 0; i < arrFName.size(); i++  )
  {
    fn = arrFName[i]; 
    tfn = fn;
    cout << i << " " << fn << endl;
    infcsv.open(fn);
    if(infcsv)
    {
      cnt = 0;
      nBOU = 0;
      nBRW = 0;
      nBSL = 0;
      nCMO = 0;
      nDED = 0;
      nFRD = 0;
      nFRN = 0;
      nGUA = 0;
      nHON = 0;
      nNEW = 0;
      nSHU = 0;
      nSIT = 0;
      nSJG = 0;
      nTUC= 0;

      while(!infcsv.eof()) 
      {
        getline(infcsv, ln);
        tln = ln;
        if(tln.empty())
          break;
        if(cnt == 0)
        {
          //cout << tln << endl; //header record
          ;
        }
        else
        {
          strcpy(szBuff, tln.c_str()); 
          pStr = strtok(szBuff, ",");
          if(pStr)
          {
            pStr = strtok(NULL, ",");
            if(pStr == NULL)
              break;
          }

          if(strcmp(pStr, "BOU") == 0)
          {
            nBOU++;
            arrBOU.push_back(ln);
          }
          if(strcmp(pStr, "SIT") == 0)
          {
            nSIT++;
            arrSIT.push_back(ln);
          }
          if(strcmp(pStr, "FRN") == 0)
          {
            nFRN++;
            arrFRN.push_back(ln);
          }
          if(strcmp(pStr, "NEW") == 0)
          {
            nNEW++;
            arrNEW.push_back(ln);
          }
          if(strcmp(pStr, "FRD") == 0)
          {
            nFRD++;
            arrFRD.push_back(ln);
          }
          if(strcmp(pStr, "DED") == 0)
          {
            nDED++;
            arrDED.push_back(ln);
          }
          if(strcmp(pStr, "BRW") == 0)
          {
            nBRW++;
            arrBRW.push_back(ln);
          }
          if(strcmp(pStr, "CMO") == 0)
          {
            nCMO++;
            arrCMO.push_back(ln);
          }
          if(strcmp(pStr, "BSL") == 0)
          {
            nBSL++;
            arrBSL.push_back(ln);
          }
          if(strcmp(pStr, "GUA") == 0)
          {
            nGUA++;
            arrGUA.push_back(ln);
          }
          if(strcmp(pStr, "HON") == 0)
          {
            nHON++;
            arrHON.push_back(ln);
          }
          if(strcmp(pStr, "SJG") == 0)
          {
            nSJG++;
            arrSJG.push_back(ln);
          }
          if(strcmp(pStr, "TUC") == 0)
          {
            nTUC++;
            arrTUC.push_back(ln);
          }
          if(strcmp(pStr, "SHU") == 0)
          {
            nSHU++;
            arrSHU.push_back(ln);
          }
        }
        cnt++;         
        rcnt++;         
      }
      infcsv.close();
    }
    else
    {
      cout << "Error\n";
    }
  }
  cout << "Number of records processed -> " << rcnt << endl;

  cout << "Number of BOU records -> " << arrBOU.size() << endl;
  cout << "Number of BRW records -> " << arrBRW.size() << endl;
  cout << "Number of BSL records -> " << arrBSL.size() << endl;
  cout << "Number of CMO records -> " << arrCMO.size() << endl;
  cout << "Number of DED records -> " << arrDED.size() << endl;
  cout << "Number of FRD records -> " << arrFRD.size() << endl;
  cout << "Number of FRN records -> " << arrFRN.size() << endl;
  cout << "Number of GUA records -> " << arrGUA.size() << endl;
  cout << "Number of HON records -> " << arrHON.size() << endl;
  cout << "Number of NEW records -> " << arrNEW.size() << endl;
  cout << "Number of SHU records -> " << arrSHU.size() << endl;
  cout << "Number of SIT records -> " << arrSIT.size() << endl;
  cout << "Number of SJG records -> " << arrSJG.size() << endl;
  cout << "Number of TUC records -> " << arrTUC.size() << endl;

  cout << "CSV File Output\n";
  cout << "CSV BOU File Output\n";
  ofBOU.open("./BOU_all.csv");
  if(ofBOU)
  {
    //for(unsigned int i = 0; arrBOU.size(); i++)
    for(vector<string>::iterator it = arrBOU.begin(); it != arrBOU.end(); ++it)
    {
      ln = *it;
      if(!ln.empty())
        ofBOU << ln << endl;
    }
    ofBOU.close();
  }

  cout << "CSV BRW File Output\n";
  ofBRW.open("./BRW_all.csv");
  if(ofBRW)
  {
    //for(unsigned int i = 0; arrBRW.size(); i++)
    for(vector<string>::iterator it = arrBRW.begin(); it != arrBRW.end(); ++it)
    {
      ln = *it;
      ofBRW << ln << endl;
    }
  }
  cout << "CSV BSL File Output\n";
  ofBSL.open("./BSL_all.csv");
  if(ofBSL)
  {
    //for(unsigned int i = 0; arrBSL.size(); i++)
    for(vector<string>::iterator it = arrBSL.begin(); it != arrBSL.end(); ++it)
    {
      ln = *it;
      ofBSL << ln << endl;
    }
  }
  cout << "CSV CMO File Output\n";
  ofCMO.open("./CMO_all.csv");
  if(ofCMO)
  {
    //for(unsigned int i = 0; arrCMO.size(); i++)
    for(vector<string>::iterator it = arrCMO.begin(); it != arrCMO.end(); ++it)
    {
      ln = *it;
      ofCMO << ln << endl;
    }
  }
  cout << "CSV DED File Output\n";
  ofDED.open("./DED_all.csv");
  if(ofDED)
  {
    //for(unsigned int i = 0; arrDED.size(); i++)
    for(vector<string>::iterator it = arrDED.begin(); it != arrDED.end(); ++it)
    {
      ln = *it;
      ofDED << ln << endl;
    }
  }
  cout << "CSV FRD File Output\n";
  ofFRD.open("./FRD_all.csv");
  if(ofFRD)
  {
    //for(unsigned int i = 0; arrFRD.size(); i++)
    for(vector<string>::iterator it = arrFRD.begin(); it != arrFRD.end(); ++it)
    {
      ln = *it;
      ofFRD << ln << endl;
    }
  }
  cout << "CSV FRN File Output\n";
  ofFRN.open("./FRN_all.csv");
  if(ofFRN)
  {
    //for(unsigned int i = 0; arrFRN.size(); i++)
    for(vector<string>::iterator it = arrFRN.begin(); it != arrFRN.end(); ++it)
    {
      ln = *it;
      ofFRN << ln << endl;
    }
  }
  cout << "CSV GUA File Output\n";
  ofGUA.open("./GUA_all.csv");
  if(ofGUA)
  {
    //for(unsigned int i = 0; arrGUA.size(); i++)
    for(vector<string>::iterator it = arrGUA.begin(); it != arrGUA.end(); ++it)
    {
      ln = *it;
      ofGUA << ln << endl;
    }
  }
  cout << "CSV HON File Output\n";
  ofHON.open("./HON_all.csv");
  if(ofHON)
  {
    //for(unsigned int i = 0; arrHON.size(); i++)
    for(vector<string>::iterator it = arrHON.begin(); it != arrHON.end(); ++it)
    {
      ln = *it;
      ofHON << ln << endl;
    }
  }
  cout << "CSV NEW File Output\n";
  ofNEW.open("./NEW_all.csv");
  if(ofNEW)
  {
    //for(unsigned int i = 0; arrNEW.size(); i++)
    for(vector<string>::iterator it = arrNEW.begin(); it != arrNEW.end(); ++it)
    {
      ln = *it;
      ofNEW << ln << endl;
    }
  }
  cout << "CSV SHU File Output\n";
  ofSHU.open("./SHU_all.csv");
  if(ofSHU)
  {
    //for(unsigned int i = 0; arrSHU.size(); i++)
    for(vector<string>::iterator it = arrSHU.begin(); it != arrSHU.end(); ++it)
    {
      ln = *it;
      ofSHU << ln << endl;
    }
  }
  cout << "CSV SIT File Output\n";
  ofSIT.open("./SIT_all.csv");
  if(ofSIT)
  {
    //for(unsigned int i = 0; arrSIT.size(); i++)
    for(vector<string>::iterator it = arrSIT.begin(); it != arrSIT.end(); ++it)
    {
      ln = *it;
      ofSIT << ln << endl;
    }
  }
  cout << "CSV SJG File Output\n";
  ofSJG.open("./SJG_all.csv");
  if(ofSJG)
  {
    //for(unsigned int i = 0; arrSJG.size(); i++)
    for(vector<string>::iterator it = arrSJG.begin(); it != arrSJG.end(); ++it)
    {
      ln = *it;
      ofSJG << ln << endl;
    }
  }
  cout << "CSV TUC File Output\n";
  ofTUC.open("./TUC_all.csv");
  if(ofTUC)
  {
    //for(unsigned int i = 0; arrTUC.size(); i++)
    for(vector<string>::iterator it = arrTUC.begin(); it != arrTUC.end(); ++it)
    {
      ln = *it;
      ofTUC << ln << endl;
    }
  }
  cout << "Done!\n";

  return 0;
}

