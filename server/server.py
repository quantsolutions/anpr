import cherrypy
import psycopg2
from psycopg2 import extras
import json
import base64
SCRIPT = """
<script>
    // Write on keyup event of keyword input element
    $("#search").keyup(function(){
        _this = this;
        // Show only matching TR, hide rest of them
        $.each($("#table tbody").find("tr"), function() {
            console.log($(this).text());
            if($(this).text().toLowerCase().indexOf($(_this).val().toLowerCase()) == -1)
               $(this).hide();
            else
                 $(this).show();                
        });
    }); 
 function showimg(src)
 {
     document.getElementById("imgdis").src = src;
     document.getElementById("popUpDiv").style.display = "block";
     setTimeout(function(){document.getElementById("popUpDiv").style.display = "none";},5000);
 }
</script>
"""
class AnprServer(object):
    @cherrypy.expose
    def index(self):
        f = open('resources/html/mainpage.html','r')
        self._getContent()
        page = f.read()
        f.close()
        content = self._getContent()
        html_content = ''
        for item in content:
            new_dict = dict(item)
            new_dict['source'] =''
            new_dict['source'] = 'data:image/jpeg;base64,'+json.loads(item['jsondata'])["jsondata"]

            html_content+= """<tr>
               <td>{numberplate}</td>
               <td>{camlocation}</td>
               <td>{time_received}</td>
               <td><img onclick="showimg('{source}')" height="48" width="64" src='{source}'/></a> </td>
               </tr>""".format(**new_dict)
        page = page.format(**{'content':html_content})
        page += SCRIPT
        return page

    @cherrypy.expose
    def pushImage(self,data):
        data = json.loads(data)
        insert_dict = {'numberplate':data['numberplate'],
                       'camlocation':data['camlocation'],
                       'jsondata':json.dumps(data)}
        result = self.cur.execute("insert into anpr.numberplates (numberplate,camlocation,jsondata,time_received) VALUES (%(numberplate)s,%(camlocation)s,%(jsondata)s,now()) ",insert_dict)
        self.conn.commit()
        return json.dumps(True)

    def _getContent(self):
        self.cur.execute('select * from anpr.numberplates order by id asc')
        return self.cur.fetchall()

    def setDBCur(self,cur):
        self.cur = cur

    def setDBConn(self,conn):
        self.conn = conn

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
    app.setDBConn(conn)
    cherrypy.quickstart(app, '/', {'/': {'tools.gzip.on': True},'/js':{'tools.staticdir.on': True,'tools.staticdir.dir': 'js'},'/html':{'tools.staticdir.on': True,'tools.staticdir.dir': 'html'}})
