# anpr
The ANPR(Automatic Number Palte Recognizer) is a opensource project created to assist local communities to monitor the traffic entering and leaving their community.

The current scope of the project is to capture vehicles and log each numberplate into the ANPR database which alows for look ups at later stages.

Current Features
================
- Python 2.7
- Potgres 9.4
- Video capturing for IP Cameras
- Proccesing of Still Images and retreval of Numberplates
- Saving of data to a server
- Cherrypy Server serving up search results


Basic Examples and use command
==============================
Working examples can be found in the examples direcotry. 

Project Structure
=================
Teh folder structure is as follows. The project is split up into 4 major folder namely db,example,client_camera and server.
The db folder contains all related database files.
The folder client_camera contains all camera aquisition as well as image proccesing files.
The folder server contains all files relating to serving the ANPR data to a webservice.
The examples folder contains examples on how to run the code.

├── camera_client
│   ├── averages
│   │   ├── 0.png
│   │   ├── 0.txt
│   │   ├── 1.png
│   │   ├── 1.txt
│   │   ├── 2.png
│   │   ├── 2.txt
│   │   ├── 3.png
│   │   ├── 3.txt
│   │   ├── 4.png
│   │   ├── 4.txt
│   │   ├── 5.png
│   │   ├── 5.txt
│   │   ├── 6.png
│   │   ├── 6.txt
│   │   ├── 7.png
│   │   ├── 7.txt
│   │   ├── 8.png
│   │   ├── 8.txt
│   │   ├── 9.png
│   │   ├── 9.txt
│   │   ├── A.png
│   │   ├── A.txt
│   │   ├── B.png
│   │   ├── B.txt
│   │   ├── C.png
│   │   ├── C.txt
│   │   ├── D.png
│   │   ├── D.txt
│   │   ├── E.png
│   │   ├── E.txt
│   │   ├── F.png
│   │   ├── F.txt
│   │   ├── G.png
│   │   ├── G.txt
│   │   ├── H.png
│   │   ├── H.txt
│   │   ├── I.png
│   │   ├── I.txt
│   │   ├── J.png
│   │   ├── J.txt
│   │   ├── K.png
│   │   ├── K.txt
│   │   ├── L.png
│   │   ├── L.txt
│   │   ├── M.png
│   │   ├── M.txt
│   │   ├── N.png
│   │   ├── N.txt
│   │   ├── O.png
│   │   ├── O.txt
│   │   ├── P.png
│   │   ├── P.txt
│   │   ├── R.png
│   │   ├── R.txt
│   │   ├── S.png
│   │   ├── S.txt
│   │   ├── T.png
│   │   ├── T.txt
│   │   ├── U.png
│   │   ├── U.txt
│   │   ├── V.png
│   │   ├── V.txt
│   │   ├── W.png
│   │   ├── W.txt
│   │   ├── X.png
│   │   ├── X.txt
│   │   ├── Y.png
│   │   ├── Y.txt
│   │   ├── Z.png
│   │   └── Z.txt
│   ├── camera.py
│   ├── image_success
│   ├── images_unprocessed
│   ├── main.py
│   ├── main.py~
│   ├── n191.net
│   ├── n251.net
│   ├── nanpr.py
│   ├── README.md
│   ├── test_data
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   └── 6.jpg
│   ├── tophat.py
│   └── tophat.py~
├── db
│   └── ANPR.db
├── examples
├── README.md
└── server
    ├── resources
    │   ├── html
    │   │   ├── login.html
    │   │   ├── main.js
    │   │   ├── mainpage.html
    │   │   └── mainpage.html~
    │   └── js
    │       └── main.js~
    ├── server.py
    └── server.py~


Depedancies
====================
sudo apt-get install python-pyfann
sudo apt-get install python-opencv
sudo apt-get install skimage
sudo apt-get install postgresql-9.4
python nanpr.py -i 2423453DR33THGP.jpg     



