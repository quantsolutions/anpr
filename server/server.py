import cherrypy
import psycopg2
from psycopg2 import extras
import json

class AnprServer(object):
    @cherrypy.expose
    def index(self):
        f = open('resources/html/mainpage.html','r')
        self._getContent()
        page = f.read()
        #print page
        return page

    @cherrypy.expose
    def pushImage(self,data):
        print ("YAYAA",data)
        insert_dict = {'numberplate':data['numberplate'],
                       'camlocation':data['camlocation'],
                       'jsondata':json.dumps(data)}
        print (insert_dict)
        result = self.cur.execute("insert into anpr.numberplates (numberplate,camlocation,jsondata,time_received) VALUES (%(numberplate)s,%(camloaction)s,%(jsondata)s,now()) ",insert_dict)
        print result
        return True

    def _getContent(self):
        result = self.cur.execute('select * from anpr.numberplates order by id asc')
        print result 

    def setDBCur(self,cur):
        self.cur = cur

if __name__ == '__main__':
   dict_cur =None
   try:
       conn = psycopg2.connect("dbname='ANPR' user='anpr_user' host='localhost' password='anpr' port='5434'")
       dict_cur = conn.cursor(cursor_factory=extras.DictCursor)
   except Exception as e:
       print "I am unable to connect to the database"
       print e
   app = AnprServer()
   app.setDBCur(dict_cur)
   cherrypy.quickstart(app, '/', {'/': {'tools.gzip.on': True}})
