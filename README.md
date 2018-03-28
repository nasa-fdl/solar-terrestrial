Frontier Development Lab 2017
B-Sting Solar Terrestrial Interactions Neural Network Generation
================================================================

**Citing:** If you use this framework please cite as follows:

    @inproceedings{2017AGUFMSM23A2591C,
        author = {{Cheung}, C.~M.~M. and {Handmer}, C. and {Kosar}, B. and {Gerules}, G. and 
	{Poduval}, B. and {Mackintosh}, G. and {Munoz-Jaramillo}, A. and 
	{Bobra}, M. and {Hernandez}, T. and {McGranaghan}, R.~M.},
        booktitle = {AGU Fall Meeting},
        title = {Modeling Geomagnetic Variations using a Machine Learning Framework},
        address = {Long Beach},
        year = {2017}
    }


Hardware Requirements - Preliminary List
---------------------
This was tested on P100 nVidia GPU and nVidia GTX 1060 GPU.  Should work if enough RAM on computer and fast enough GPU.

Software Requirements
------------
python, anaconda, keras-gp, scikit-learn, pandas

How to run
-----

For some of the csv files c++ programs were used for conversion.  They can be found in cpp_files subdirectory.  These are placeholder programs and probably should be converted to python for c++ phobic people.  Check the README in that subdirectory for further details.  

For either Project 1 or Project 2 edit the config.cfg file to point to the appropriate directory and import data.

Project 1: geomag with omni solar wind data
Two files to run to explore LSTM with geo magnetic and solar wind data.  Each one is a different take on LSTMs.

python LSTMarrayprediction.py

or 

python lstm_multi_channel.py

Project 2: geomag, omni solar wind data and kp index

python kp_regress.py


Contact Information: 

Mark Cheung cheung@lmsal.com

George Gerules ggerules@gmail.com or gwgkt2@mail.umsl.edu

